import math
import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.gqa.experiment import run


def per_head_oracle(q, k, v, mask=None):
    """Independent group selection and library attention, including native VJPs."""
    h, g = q.shape[1], k.shape[1]
    return torch.cat([F.scaled_dot_product_attention(q[:, i:i+1], k[:, i // (h // g):i // (h // g)+1],
        v[:, i // (h // g):i // (h // g)+1], attn_mask=mask) for i in range(h)], dim=1)


@pytest.mark.t0
def test_shapes_precision_strides_and_input_preservation(impl):
    """Different query/value widths, noncontiguous inputs and empty queries retain the public layout."""
    for dtype in (torch.float32, torch.float64):
        for batch, queries in ((2, 3), (0, 3), (1, 0)):
            q = torch.randn(batch, 4, queries, 6, dtype=dtype)[..., ::2]
            k = torch.randn(batch, 2, 5, 6, dtype=dtype)[..., ::2]
            v = torch.randn(batch, 2, 5, 4, dtype=dtype)[..., ::2]
            before = [item.clone() for item in (q, k, v)]
            output, weights = impl.grouped_query_attention(q, k, v)
            assert output.shape == (batch, 4, queries, 2) and weights.shape == (batch, 4, queries, 5)
            assert output.dtype == weights.dtype == dtype and output.device == q.device
            for a, b in zip((q, k, v), before): close(a, b)
        projection = torch.randn(8, 3, 6, dtype=dtype)[..., ::2]
        before = projection.clone()
        pooled = impl.mean_pool_kv_heads(projection, 2)
        assert pooled.shape == (2, 3, 3) and pooled.dtype == dtype and pooled.device == projection.device
        close(projection, before)


@pytest.mark.t1
def test_hand_computed_groups_and_projection_means(impl):
    """Four query heads form [0,0,1,1] KV groups; zero queries uniformly average their group's values."""
    q, k = torch.zeros(1, 4, 1, 2, dtype=torch.float64), torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    v = torch.tensor([[[[2.], [6.]], [[10.], [30.]]]], dtype=q.dtype)
    output, weights = impl.grouped_query_attention(q, k, v)
    close(output, torch.tensor([[[[4.]], [[4.]], [[20.]], [[20.]]]], dtype=q.dtype), atol=1e-12, rtol=1e-12)
    close(weights, torch.full((1, 4, 1, 2), .5, dtype=q.dtype), atol=1e-12, rtol=1e-12)
    projection = torch.tensor([[[0., 2.]], [[2., 4.]], [[10., 14.]], [[14., 18.]]], dtype=q.dtype)
    close(impl.mean_pool_kv_heads(projection, 2), torch.tensor([[[1., 3.]], [[12., 16.]]], dtype=q.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_contiguous_routing_and_scalar_softmax(impl):
    """A scalar dot-product oracle checks every returned weight and exposes cyclic grouping or wrong scaling."""
    q, k, v = [torch.randn(*shape, dtype=torch.float64) for shape in ((1, 6, 2, 2), (1, 3, 3, 2), (1, 3, 3, 1))]
    output, weights = impl.grouped_query_attention(q, k, v)
    for h in range(6):
        group = h // 2
        for t in range(2):
            scores = [sum(q[0, h, t, d].item() * k[0, group, j, d].item() for d in range(2)) / math.sqrt(2) for j in range(3)]
            exp = [math.exp(value - max(scores)) for value in scores]
            probabilities = [value / sum(exp) for value in exp]
            close(weights[0, h, t], torch.tensor(probabilities, dtype=q.dtype), atol=1e-12, rtol=1e-12)
            expected = sum(p * v[0, group, j, 0].item() for j, p in enumerate(probabilities))
            close(output[0, h, t, 0], torch.tensor(expected, dtype=q.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_mha_mqa_endpoints_and_group_permutation(impl):
    """G=1 recovers MQA, G=H recovers MHA; a consistent group permutation only permutes outputs."""
    q = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    for groups in (1, 2, 4):
        k, v = torch.randn(2, groups, 5, 3, dtype=q.dtype), torch.randn(2, groups, 5, 2, dtype=q.dtype)
        output, _ = impl.grouped_query_attention(q, k, v)
        close(output, per_head_oracle(q, k, v), atol=1e-12, rtol=1e-11)
        if groups == 2:
            order = torch.tensor([2, 3, 0, 1])
            swapped, _ = impl.grouped_query_attention(q[:, order], k.flip(1), v.flip(1))
            close(swapped, output[:, order], atol=1e-12, rtol=1e-11)


@pytest.mark.t2
def test_boolean_masks_zero_rows_and_causal_isolation(impl):
    """Boolean visibility broadcasts over groups; padded rows stay zero and future KV changes cannot leak."""
    q, k, v = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in ((1, 4, 3, 2), (1, 2, 3, 2), (1, 2, 3, 3))]
    visible = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    output, weights = impl.grouped_query_attention(q, k, v, visible)
    close(output[:, :, 0], torch.zeros_like(output[:, :, 0]))
    close(weights[:, :, 0], torch.zeros_like(weights[:, :, 0]))
    for gradient in torch.autograd.grad(output.sum(), (q, k, v), allow_unused=True):
        assert gradient is not None and torch.isfinite(gradient).all()
    padded_output, _ = impl.grouped_query_attention(q, k, v, visible)
    for gradient in torch.autograd.grad(padded_output[:, :, 0].sum(), (q, k, v), allow_unused=True):
        assert gradient is not None
        close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, :, 2] += 100; changed_v[:, :, 2] -= 200
    changed, _ = impl.grouped_query_attention(q, changed_k, changed_v, visible)
    close(changed[:, :, :2], output[:, :, :2], atol=1e-12, rtol=1e-12)
    head_mask = visible[None, None].expand(1, 4, 3, 3).clone(); head_mask[:, 1] = False
    masked, _ = impl.grouped_query_attention(q, k, v, head_mask)
    close(masked[:, 1], torch.zeros_like(masked[:, 1]))


@pytest.mark.t2
def test_all_qkv_gradients_accumulate_across_shared_heads(impl):
    """Each shared KV gradient must equal the sum of independent per-query-head library gradients."""
    for groups in (1, 2, 4):
        tensors = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in ((2, 4, 2, 3), (2, groups, 3, 3), (2, groups, 3, 2))]
        copies = [value.detach().clone().requires_grad_() for value in tensors]
        actual, _ = impl.grouped_query_attention(*tensors)
        expected = per_head_oracle(*copies)
        upstream = torch.randn_like(actual)
        gradients = torch.autograd.grad((actual * upstream).sum(), tensors, allow_unused=True)
        for a, b in zip(gradients, torch.autograd.grad((expected * upstream).sum(), copies)):
            assert a is not None, 'Shared KV tensors must stay connected to the graph'
            close(a, b, atol=1e-11, rtol=1e-10)


@pytest.mark.t2
def test_projection_pooling_endpoints_and_analytical_gradient(impl):
    """Conversion averages only head groups; every original head receives 1/group_size of its pooled gradient."""
    weight = torch.randn(8, 3, 10, dtype=torch.float64)[..., ::2].detach().requires_grad_()
    for count in (1, 2, 4, 8):
        size = 8 // count
        output = impl.mean_pool_kv_heads(weight, count)
        expected = torch.stack([sum((weight[h] for h in range(i * size, (i + 1) * size))) / size for i in range(count)])
        close(output, expected, atol=1e-12, rtol=1e-12)
        upstream = torch.randn_like(output)
        gradient, = torch.autograd.grad((output * upstream).sum(), (weight,))
        close(gradient, torch.stack([upstream[h // size] / size for h in range(8)]), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_small_numerical_jacobians(impl):
    """Finite differences independently cover attention outputs, returned weights and checkpoint pooling."""
    q, k, v = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in ((1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 2, 1))]
    assert torch.autograd.gradcheck(lambda *args: impl.grouped_query_attention(*args), (q, k, v), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)
    weight = torch.randn(2, 1, 2, dtype=q.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: impl.mean_pool_kv_heads(w, 1), (weight,), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)


@pytest.mark.t2
def test_invalid_head_counts_and_dimensions(impl):
    """Reject unequal/nonintegral groups instead of silently routing or truncating query heads."""
    weight = torch.randn(4, 3, 2)
    for invalid in (0, -1, 3, 5, True, False, 1.5, '2', None):
        with pytest.raises(ValueError): impl.mean_pool_kv_heads(weight, invalid)
    q, k, v = torch.randn(1, 4, 2, 3), torch.randn(1, 2, 5, 3), torch.randn(1, 2, 5, 2)
    for args in ((q[:, :3], k, v), (q, k[:, :0], v[:, :0]), (q[:, :0], k, v),
                 (q, k, v[:, :1]), (q, k, v[:, :, :4]), (q, k[..., :2], v),
                 (q[0], k, v), (q, k[:, :, :0], v[:, :, :0])):
        with pytest.raises(ValueError): impl.grouped_query_attention(*args)


@pytest.mark.t3
def test_cache_capacity_and_conversion_experiment(impl, artifact):
    """All four group counts run the learner's operators; stored KV ratios are exact and MHA is the unchanged endpoint."""
    result = run(impl.mean_pool_kv_heads, impl.grouped_query_attention)
    artifact('gqa_conversion_experiment', result)
    metrics = result['metrics']
    assert metrics['maximum_attention_oracle_error'] < 1e-10
    assert metrics['mha_endpoint_rmse'] < 1e-12
    assert metrics['mqa_conversion_rmse'] > 1e-3
    assert metrics['mqa_cache_ratio'] == .125 and metrics['gqa2_cache_ratio'] == .25
    assert metrics['mha_cache_elements'] == 240
