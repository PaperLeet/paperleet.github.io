# PagedAttention solution

Compute the required page count as `(L + S - 1) // S`. Select that prefix of the block table, gather the physical cache pages, flatten the page and slot axes, then retain the first $L$ tokens. Sorting page IDs or reading the first physical pages would change the logical context.

For each request, form query-key dot products per head and token, then divide by $\sqrt D$. Apply softmax along the token axis. Weight the values and sum along that same axis to produce `(H,D)`; stack requests into `(B,H,D)`.

All pages share one normalization denominator. Computing a separate softmax per page assigns incorrect total weight to each page. The reference gathers a logical sequence to make this specification explicit; a GPU implementation can instead read blocks and combine normalization statistics.

Truncate the final page before validating active values or calculating attention. NaN and large values outside the valid context must never reach the attention denominator. Unused table entries need not contain valid page IDs.

Indexing and reshaping preserve autograd. Shared physical pages accumulate contributions from every request that reads them. With zero queries and keys, weights are uniform: a value token shared by contexts of lengths 3 and 4 receives gradient $1/3+1/4$ per feature when all outputs are summed. Unread slots receive zero gradient.

The replay experiment checks layout-independent results as context lengths cross page boundaries. Its storage calculation counts distinct referenced physical pages, counting a shared prefix only once. It is an accounting example rather than a speed benchmark.
