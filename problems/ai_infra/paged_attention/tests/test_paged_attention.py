import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.paged_attention.experiment import run


def _oracle(query, key_cache, value_cache, tables, lengths):
    result = []
    for b, length in enumerate(lengths.tolist()):
        heads = []
        for h, q in enumerate(query[b].tolist()):
            keys, values = [], []
            for token in range(length):
                page = int(tables[b, token // key_cache.shape[1]])
                slot = token % key_cache.shape[1]
                keys.append(key_cache[page, slot, h].tolist())
                values.append(value_cache[page, slot, h].tolist())
            scores = [sum(x*y for x, y in zip(q, k)) / math.sqrt(len(q)) for k in keys]
            maximum = max(scores)
            weights = [math.exp(x-maximum) for x in scores]
            heads.append([sum(w*v[d] for w, v in zip(weights, values)) / sum(weights) for d in range(len(q))])
        result.append(heads)
    return query.new_tensor(result)


@pytest.mark.t0
def test_shapes_dtype_and_unchanged_storage(impl):
    """Gathered contexts and batched outputs preserve dtype/device without editing the cache."""
    for dtype in (torch.float32, torch.float64):
        key, value = torch.randn(5, 3, 2, 4, dtype=dtype), torch.randn(5, 3, 2, 4, dtype=dtype)
        query = torch.randn(2, 2, 4, dtype=dtype)
        tables = torch.tensor([[3, 0, -1], [3, 4, 1]])
        lengths = torch.tensor([4, 7])
        saved = [x.clone() for x in (key, value, query, tables, lengths)]
        k, v = impl.gather_paged_kv(key, value, tables[0], 4)
        output = impl.paged_attention(query, key, value, tables, lengths)
        assert k.shape == v.shape == (4, 2, 4) and output.shape == query.shape
        assert k.dtype == v.dtype == output.dtype == dtype and output.device == query.device
        for x, before in zip((key, value, query, tables, lengths), saved): close(x, before)


@pytest.mark.t0
def test_reject_invalid_active_tables_and_empty_context(impl):
    """Active IDs, table dtypes, head shapes and positive lengths are validated; unused IDs are ignored."""
    key = torch.ones(3, 2, 1, 2); table = torch.tensor([2, 0, -1])
    for length in (0, -1, 7, True, 2.5):
        with pytest.raises(ValueError): impl.gather_paged_kv(key, key, table, length)
    for invalid in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([0.])):
        with pytest.raises(ValueError): impl.gather_paged_kv(key, key, invalid, 1)
    with pytest.raises(ValueError): impl.gather_paged_kv(key, key.double(), table, 2)
    with pytest.raises(ValueError): impl.paged_attention(torch.ones(1, 2, 2), key, key, table[None], torch.tensor([2]))
    with pytest.raises(ValueError): impl.paged_attention(torch.ones(1, 1, 2), key, key, table[None], torch.tensor([2.]))
    k, _ = impl.gather_paged_kv(key, key, table, 3)
    assert k.shape == (3, 1, 2)


@pytest.mark.t1
def test_noncontiguous_page_order_and_partial_tail(impl):
    """Logical positions use table IDs and stop inside the last page, never sorting physical pages."""
    key = torch.arange(6 * 3 * 2, dtype=torch.float64).reshape(6, 3, 1, 2)
    value = key * -.7 + 2
    table = torch.tensor([4, 1, 5, -1])
    k, v = impl.gather_paged_kv(key, value, table, 7)
    expected = torch.stack([key[4, 0], key[4, 1], key[4, 2], key[1, 0], key[1, 1], key[1, 2], key[5, 0]])
    close(k, expected); close(v, expected * -.7 + 2)


@pytest.mark.t1
def test_batched_attention_against_scalar_oracle(impl):
    """Independent scalar dot products and exponentials check heads, lengths, scale and value weighting."""
    key, value = torch.randn(7, 2, 3, 4, dtype=torch.float64), torch.randn(7, 2, 3, 4, dtype=torch.float64)
    query = torch.randn(3, 3, 4, dtype=torch.float64)
    tables = torch.tensor([[4, 0, 3], [4, 6, -1], [2, -1, -1]])
    lengths = torch.tensor([5, 3, 1])
    close(impl.paged_attention(query, key, value, tables, lengths),
          _oracle(query, key, value, tables, lengths), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_physical_relocation_is_invariant(impl):
    """Moving every physical page and remapping the table preserves logical attention exactly."""
    key, value = torch.randn(5, 2, 2, 3, dtype=torch.float64), torch.randn(5, 2, 2, 3, dtype=torch.float64)
    query = torch.randn(2, 2, 3, dtype=torch.float64)
    table = torch.tensor([[4, 0, 2], [4, 1, 3]]); lengths = torch.tensor([5, 6])
    permutation = torch.tensor([2, 4, 1, 0, 3]); inverse = torch.argsort(permutation)
    before = impl.paged_attention(query, key, value, table, lengths)
    after = impl.paged_attention(query, key[permutation], value[permutation], inverse[table], lengths)
    close(before, after, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_padding_and_unused_pages_are_invisible(impl):
    """Unused table entries, unallocated pages and final-page tail values cannot affect a request."""
    key, value = torch.randn(4, 3, 1, 2, dtype=torch.float64), torch.randn(4, 3, 1, 2, dtype=torch.float64)
    query = torch.randn(1, 1, 2, dtype=torch.float64)
    table = torch.tensor([[2, 0, -1]]); lengths = torch.tensor([4])
    expected = impl.paged_attention(query, key, value, table, lengths)
    key[0, 1:] = float('nan'); value[0, 1:] = float('nan')
    key[1] = float('nan'); value[1] = float('nan')
    key[3] = 1e20; value[3] = -1e20
    table[0, 2] = 999
    close(impl.paged_attention(query, key, value, table, lengths), expected)


@pytest.mark.t2
def test_shared_prefix_gradients_accumulate_and_tail_stays_zero(impl):
    """Shared physical tokens receive summed contributions while padding and unused pages receive no gradient."""
    key = torch.zeros(4, 2, 1, 2, dtype=torch.float64, requires_grad=True)
    value = torch.arange(16, dtype=torch.float64).reshape(4, 2, 1, 2).requires_grad_()
    query = torch.zeros(2, 1, 2, dtype=torch.float64, requires_grad=True)
    table = torch.tensor([[2, 0], [2, 1]]); lengths = torch.tensor([3, 4])
    output = impl.paged_attention(query, key, value, table, lengths)
    dq, dk, dv = torch.autograd.grad(output.sum(), (query, key, value))
    expected = torch.zeros_like(value)
    expected[2] = 1/3 + 1/4
    expected[0, 0] = 1/3
    expected[1] = 1/4
    close(dv, expected, atol=1e-12, rtol=1e-12)
    close(dq, torch.zeros_like(query)); close(dk, torch.zeros_like(key))


@pytest.mark.t2
def test_query_and_key_gradients_against_finite_difference(impl):
    """Numerical derivatives validate differentiable attention over scattered cache locations."""
    query = torch.tensor([[[.3, -.7]]], dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[[[.2, .9]], [[-.5, .4]]], [[[.8, -.3]], [[.6, .2]]]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[[[2., -.3]], [[.7, .8]]], [[[-1., .4]], [[.2, -.5]]]], dtype=torch.float64)
    table = torch.tensor([[1, 0]]); lengths = torch.tensor([3])
    grads = torch.autograd.grad(impl.paged_attention(query, key, value, table, lengths).sum(), (query, key))
    for index, tensor in enumerate((query, key)):
        expected = torch.zeros_like(tensor)
        for coordinate in range(tensor.numel()):
            plus = [query.detach().clone(), key.detach().clone()]
            minus = [query.detach().clone(), key.detach().clone()]
            plus[index].view(-1)[coordinate] += 1e-6
            minus[index].view(-1)[coordinate] -= 1e-6
            upper = _oracle(plus[0], plus[1], value, table, lengths).sum()
            lower = _oracle(minus[0], minus[1], value, table, lengths).sum()
            expected.view(-1)[coordinate] = (upper-lower) / 2e-6
        close(grads[index], expected, atol=1e-8, rtol=1e-6)


@pytest.mark.t3
def test_decode_replay_and_slot_accounting(impl, artifact):
    """Growing requests match scalar dense attention across block boundaries while sharing one prefix page."""
    result = run(impl.gather_paged_kv, impl.paged_attention)
    artifact('paged_decode_replay', result)
    m = result['metrics']
    assert m['maximum_attention_error'] < 1e-12 and m['maximum_gather_error'] == 0
    assert m['initial_paged_slots'] == 3
    assert m['final_paged_slots'] == 15 < m['reserved_contiguous_slots']
    assert m['shared_prefix_slots_saved'] == 3
