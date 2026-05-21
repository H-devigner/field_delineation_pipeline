"""
IDMapper — Union-Find for globally consistent field ID mapping.

Direct port from Delineate-Anything/methods/main/IDMapper.py.

When tiles overlap, the same real-world field may get different IDs
in different tiles. The incremental union-find tracks which IDs
should merge, and `finalize()` returns a mapping array so the
instance raster can be remapped with a single numpy indexing op:

    mapper.union([id_a, id_b])
    parent = mapper.finalize()
    instances[:] = parent[instances]
"""

import numpy as np


class IncrementalFastMapper:
    """Incremental union-find with path compression for ID merging."""

    def __init__(self, initial_size: int = 10_000_000):
        self.parent = np.arange(initial_size, dtype=np.int32)
        self.capacity = initial_size
        self.dirty: set = set()

    def _ensure_capacity(self, max_id: int):
        if max_id < self.capacity:
            return
        new_capacity = max_id + 1_000_000
        new_parent = np.arange(new_capacity, dtype=np.int32)
        new_parent[:self.capacity] = self.parent
        self.parent = new_parent
        self.capacity = new_capacity

    def find(self, x: int) -> int:
        """Find root with path compression."""
        self._ensure_capacity(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while x != root:
            p = self.parent[x]
            self.parent[x] = root
            x = p
        return root

    def union(self, ids):
        """Merge a set of IDs into one group (smallest root wins)."""
        roots = set(self.find(u) for u in ids)
        if len(roots) < 2:
            return
        rmin = min(roots)
        for r in roots:
            if r != rmin:
                self.parent[r] = rmin
                self.dirty.add(r)
        for u in ids:
            self.dirty.add(u)

    def finalize(self) -> np.ndarray:
        """Resolve all dirty entries and return the parent array."""
        for i in self.dirty:
            if i < self.capacity:
                self.parent[i] = self.find(i)
        self.dirty.clear()
        return self.parent

    def reset(self):
        """Reset to identity mapping."""
        self.parent[:] = np.arange(self.capacity, dtype=np.int32)
        self.dirty.clear()
