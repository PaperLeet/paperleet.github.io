def hybrid_batch(decodes: list[int], prefills: list[int], budget: int, chunk_size: int) -> dict:
    """Decode IDs, scheduled (prefill_index,token_count) pairs, and remaining tokens.

decodes is a list of distinct nonnegative IDs; prefills contains positive remaining token counts in FIFO order. budget>=len(decodes), chunk_size>=1. Every decode runs once. Give the head prefill min(remaining,chunk_size,budget_left); if unfinished, stop admitting later prefills. At most one chunk per request per iteration. Return decodes, prefills as (index,count), and remaining with zeros for completed items."""
    raise NotImplementedError()
