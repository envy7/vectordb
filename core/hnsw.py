"""
hnsw.py — Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.

The problem with brute force search:
    Every query scans all n vectors: O(n · d). For 1M vectors it's fine.
    For 100M it's too slow. We need approximate nearest neighbor (ANN) search.

The HNSW idea — a navigable graph with layers:
    Instead of scanning everything, build a graph where similar vectors are
    connected. Then search by graph traversal.

    The hierarchy is the key innovation:
        Layer 2 (top): 10 nodes, connected over long distances  → highway network
        Layer 1:      100 nodes, medium distances               → main roads
        Layer 0 (bottom): all nodes, short distances            → local streets

    Search starts at the top layer and greedily moves toward the query.
    At each layer it descends to the next, now starting from a much better
    position. By the time it reaches layer 0, it's already near the answer
    and only needs to explore a small local neighborhood.

Complexity:
    Construction: O(n · log n)    ← vs brute force: no construction
    Search:       O(log n)        ← vs brute force: O(n · d)
    Memory:       O(n · M)        ← M edges per node per layer

Parameters:
    M               Connections per node per layer. More = better recall but
                    more memory and slower construction. Typical: 8-64.
                    M=16 is a good default.

    ef_construction Candidate list size during construction. Larger = better
                    graph quality but slower build. Typical: 100-500.
                    ef_construction=200 is a good default.

    ef (at search)  Candidate list size during search. Larger = better recall
                    but slower query. Must be >= k. Typical: 50-200.
                    ef=50 is a good default.

Implementation follows Algorithms 1, 2, 4, 5 from the original paper:
    Malkov & Yashunin (2018): https://arxiv.org/abs/1603.09320

Further reading:
    HNSW explained visually: https://www.pinecone.io/learn/series/faiss/hnsw/
    ANN benchmarks:          https://ann-benchmarks.com/
"""

import heapq
import json
import math
import os
import random

import numpy as np


class HNSWIndex:
    """
    HNSW approximate nearest neighbor index.

    Node ids are sequential integers (0, 1, 2, ...) matching the order
    vectors are inserted. This must match VectorStorage._records order
    so that a node_id from search maps back to the correct record.
    """

    def __init__(self, M: int = 16, ef_construction: int = 200):
        self.M = M
        self.M0 = 2 * M           # layer 0 allows more connections (paper section 4.1)
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(M)  # level probability normalisation factor

        # Parallel to VectorStorage._records — stored here for distance computation
        self._vectors: list[np.ndarray] = []

        # _layers[layer][node_id] = list of neighbor node_ids
        # Layer 0 contains every node; higher layers contain exponentially fewer.
        self._layers: list[dict[int, list[int]]] = []

        self._entry_point: int | None = None  # node_id of the current top entry
        self._max_layer: int = -1             # index of the highest non-empty layer

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine distance in [0, 2]. Lower = more similar.

        We use cosine distance (not Euclidean) for the same reason as brute
        force search: embedding direction matters, not magnitude.

        After L2-normalising both vectors, this reduces to 1 - dot(a, b).
        """
        a_n = a / (np.linalg.norm(a) + 1e-10)
        b_n = b / (np.linalg.norm(b) + 1e-10)
        return float(1.0 - np.dot(a_n, b_n))

    # ------------------------------------------------------------------
    # Level sampling
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """
        Sample the maximum layer for a new node.

        Drawn from a geometric distribution: most nodes land at layer 0,
        a few reach layer 1, fewer still reach layer 2, etc.
        This gives the hierarchical structure automatically.

        Formula: floor(-ln(uniform(0,1)) / ln(M))
        """
        return int(-math.log(random.random() + 1e-10) * self.ml)

    # ------------------------------------------------------------------
    # Layer search — Algorithm 2 from the paper
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """
        Find ef approximate nearest neighbors to `query` within one layer.

        Uses two heaps for efficiency:
            candidates: min-heap (dist, node) — next nodes to explore
            found:      max-heap (-dist, node) — best ef nodes seen so far

        The early-exit condition (dist_c > worst_dist) means we stop as soon
        as every remaining candidate is already worse than our current worst
        found — the graph guarantees we won't find anything better beyond that.

        Returns: list of (dist, node_id) sorted ascending by distance.
        """
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []  # stored as (-dist, node) for max-heap

        for ep in entry_points:
            d = self._dist(query, self._vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(found, (-d, ep))

        layer_graph = self._layers[layer] if layer < len(self._layers) else {}

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_dist = -found[0][0]

            if dist_c > worst_dist:
                break  # no candidate can beat the current worst found

            for neighbor in layer_graph.get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                d_n = self._dist(query, self._vectors[neighbor])
                worst_dist = -found[0][0]

                if len(found) < ef or d_n < worst_dist:
                    heapq.heappush(candidates, (d_n, neighbor))
                    heapq.heappush(found, (-d_n, neighbor))
                    if len(found) > ef:
                        heapq.heappop(found)  # evict the farthest

        return sorted((-d, nid) for d, nid in found)

    # ------------------------------------------------------------------
    # Neighbor selection
    # ------------------------------------------------------------------

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Return the M closest node ids. Simple strategy from Algorithm 3."""
        return [nid for _, nid in sorted(candidates)[:M]]

    # ------------------------------------------------------------------
    # Insert — Algorithm 1 from the paper
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray) -> int:
        """
        Insert a vector into the index. Returns the assigned node id.

        Steps:
            1. Sample a random level l for the new node
            2. From the top layer down to l+1: greedy descent (ef=1) to find
               the best entry point for this node's level
            3. From l down to 0: find ef_construction neighbors, connect
               bidirectionally, trim connections if over capacity
            4. If l is a new maximum layer, update the entry point
        """
        node_id = len(self._vectors)
        self._vectors.append(vector.copy())

        level = self._random_level()

        # Extend layers list to accommodate this level
        while len(self._layers) <= level:
            self._layers.append({})

        # Initialise empty adjacency lists for this node
        for l in range(level + 1):
            self._layers[l][node_id] = []

        if self._entry_point is None:
            # First node — it becomes the sole entry point
            self._entry_point = node_id
            self._max_layer = level
            return node_id

        ep = [self._entry_point]

        # Phase 1: greedy descent from top to level+1 (ef=1, just finding vicinity)
        for l in range(self._max_layer, level, -1):
            if l < len(self._layers):
                results = self._search_layer(vector, ep, ef=1, layer=l)
                ep = [results[0][1]]

        # Phase 2: for each layer this node belongs to, find neighbors and connect
        for l in range(min(level, self._max_layer), -1, -1):
            M_max = self.M0 if l == 0 else self.M

            results = self._search_layer(
                vector, ep, ef=self.ef_construction, layer=l
            )
            neighbors = self._select_neighbors(results, M_max)

            # New node → neighbors
            self._layers[l][node_id] = neighbors

            # neighbors → new node (bidirectional), trim if over capacity
            for neighbor in neighbors:
                self._layers[l][neighbor].append(node_id)
                if len(self._layers[l][neighbor]) > M_max:
                    neighbor_vec = self._vectors[neighbor]
                    existing = [
                        (self._dist(neighbor_vec, self._vectors[n]), n)
                        for n in self._layers[l][neighbor]
                    ]
                    self._layers[l][neighbor] = self._select_neighbors(existing, M_max)

            ep = [nid for _, nid in results]

        # If new node sits in a higher layer, it becomes the new entry point
        if level > self._max_layer:
            self._entry_point = node_id
            self._max_layer = level

        return node_id

    # ------------------------------------------------------------------
    # Search — Algorithm 5 from the paper
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        ef: int = 50,
    ) -> list[tuple[float, int]]:
        """
        Find the k approximate nearest neighbors to `query`.

        Args:
            query : (dim,) query vector
            k     : number of results to return
            ef    : search-time candidate list size.
                    Higher = better recall, slower query.
                    Must be >= k. Rule of thumb: ef=50 gives ~95% recall.

        Returns:
            List of (cosine_similarity, node_id) sorted descending by similarity.
            cosine_similarity is in [-1, 1]; higher = more similar.
        """
        if self._entry_point is None or not self._vectors:
            return []

        ef = max(ef, k)
        ep = [self._entry_point]

        # Greedy descent through upper layers (ef=1 — just zooming in)
        for l in range(self._max_layer, 0, -1):
            if l < len(self._layers):
                results = self._search_layer(query, ep, ef=1, layer=l)
                ep = [results[0][1]]

        # Full search at layer 0 with ef candidates
        results = self._search_layer(query, ep, ef=ef, layer=0)

        # Convert cosine distance → cosine similarity; return top k
        return [(1.0 - dist, nid) for dist, nid in results[:k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Save the graph structure to <directory>/hnsw.json.

        Vectors are NOT saved here — they live in vectors.npy (VectorStorage).
        Only the graph topology and parameters are saved.
        """
        data = {
            "M":               self.M,
            "ef_construction": self.ef_construction,
            "max_layer":       self._max_layer,
            "entry_point":     self._entry_point,
            # JSON requires string keys
            "layers": [
                {str(k): v for k, v in layer.items()}
                for layer in self._layers
            ],
        }
        with open(os.path.join(directory, "hnsw.json"), "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, directory: str, vectors: list[np.ndarray]) -> "HNSWIndex":
        """
        Load graph from disk and attach the provided vector list.

        Args:
            directory : directory containing hnsw.json
            vectors   : list of np.ndarray, one per stored record,
                        in the same order as VectorStorage._records
        """
        with open(os.path.join(directory, "hnsw.json")) as f:
            data = json.load(f)

        idx = cls(M=data["M"], ef_construction=data["ef_construction"])
        idx._max_layer   = data["max_layer"]
        idx._entry_point = data["entry_point"]
        idx._vectors     = vectors  # shared reference — no duplication
        idx._layers = [
            {int(k): v for k, v in layer.items()}
            for layer in data["layers"]
        ]
        return idx

    def __len__(self) -> int:
        return len(self._vectors)
