def iteration_schedule(arrivals: list[int], lengths: list[int], capacity: int) -> tuple[list[tuple], list[int]]:
    """Per-iteration active request IDs and completion times in original order.

arrivals and lengths are equal-length integer lists; arrivals>=0, lengths>=1, capacity>=1. Break arrival ties by input index. Fill slots at each iteration boundary in (arrival,index) order without preempting active requests. Completion is the end of the final occupied slot. Return (batches, completion_times), preserving idle slots as empty tuples."""
    raise NotImplementedError()
