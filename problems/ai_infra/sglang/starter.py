def radix_cache_step(cache: dict, prompt: tuple[int, ...], capacity: int, clock: int, pinned: tuple=()) -> tuple[dict, int]:
    """Copied cache after prefix reuse/insertion/eviction, and matched prefix length.

cache maps nonempty token tuples to integer last-use times and must be prefix-closed. prompt is a nonempty tuple of nonnegative ints; capacity counts nonroot token nodes. clock is at least existing timestamps. pinned names cached nodes whose ancestors are also protected. Protect the current prompt path too; break LRU ties lexicographically by token tuple. Return (new_cache, reused_tokens), never mutating input even on failure."""
    raise NotImplementedError()
