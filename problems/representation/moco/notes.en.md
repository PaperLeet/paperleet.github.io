# Hints

1. Normalize query and detached key/queue rows.
2. Put paired positive dots first, followed by query-dictionary products; keep every column in logsumexp.
3. EMA weights the old key by m. Return detached independent storage.
4. Normalize new keys and write row i into `(pointer+i)%K`; return the advanced pointer.
5. Encode keys before query SGD and key EMA; enqueue the cached keys afterward.
