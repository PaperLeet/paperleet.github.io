"""Independent NumPy equations, directional derivatives and MHA lifecycle."""
import copy
import itertools
import pickle

import numpy as np
import pytest

from stridetorch import as_torch, torch


def reference(inputs, state, *, heads=2, layout='sequence', padding=None, mask=None,
              bias_tokens=False, zero_token=False, average=True, need=True,
              static=None, dropout_multiplier=None, causal=False):
    """Project each operand independently; explicit N,H,L,D contractions."""
    q, k, v = inputs
    unbatched = layout == 'unbatched'
    if unbatched:
        q, k, v = (a[:, None] for a in (q, k, v))
    elif layout == 'batch':
        q, k, v = (a.swapaxes(0, 1) for a in (q, k, v))
    length, batch, embed = q.shape
    width = embed // heads
    weights = [state[n + '_proj_weight'] for n in 'qkv'] if 'in_proj_weight' not in state else np.split(state['in_proj_weight'], 3)
    biases = np.split(state['in_proj_bias'], 3) if 'in_proj_bias' in state else [0, 0, 0]
    q, k, v = [a @ w.T + b for a, w, b in zip((q, k, v), weights, biases)]
    if mask is not None:
        mask = np.where(mask, -np.inf, 0).astype(q.dtype) if mask.dtype.kind == 'b' else mask.copy()
        if mask.ndim == 2: mask = mask[None, None]
        else: mask = mask.reshape(batch, heads, length, -1)
    if padding is not None:
        padding = np.where(padding, -np.inf, 0).astype(q.dtype) if padding.dtype.kind == 'b' else padding.copy()
        padding = padding.reshape(batch, 1, 1, -1)
    if bias_tokens:
        k = np.concatenate([k, np.broadcast_to(state['bias_k'], (1, batch, embed))], axis=0)
        v = np.concatenate([v, np.broadcast_to(state['bias_v'], (1, batch, embed))], axis=0)
        if mask is not None: mask = np.pad(mask, [(0, 0)] * 3 + [(0, 1)])
        if padding is not None: padding = np.pad(padding, [(0, 0)] * 3 + [(0, 1)])
    q, k, v = [a.reshape(a.shape[0], batch, heads, width).transpose(1, 2, 0, 3) for a in (q, k, v)]
    if static is not None:
        k, v = [a.reshape(batch, heads, a.shape[1], width) for a in static]
    if zero_token:
        k, v = [np.concatenate([a, np.zeros((batch, heads, 1, width), dtype=a.dtype)], axis=2) for a in (k, v)]
        if mask is not None: mask = np.pad(mask, [(0, 0)] * 3 + [(0, 1)])
        if padding is not None: padding = np.pad(padding, [(0, 0)] * 3 + [(0, 1)])
    scores = np.einsum('nhld,nhsd->nhls', q, k) / np.sqrt(width)
    if causal:
        scores = np.where(np.arange(k.shape[2])[None, :] <= np.arange(length)[:, None], scores, -np.inf)
    elif mask is not None: scores = scores + mask
    if padding is not None: scores = scores + padding
    if scores.shape[-1]:
        with np.errstate(invalid='ignore', divide='ignore'):
            maximum = scores.max(-1, keepdims=True)
            if not need: maximum = np.where(np.isneginf(maximum), 0, maximum)
            exponent = np.exp(scores - maximum)
            denominator = exponent.sum(-1, keepdims=True)
            probability = exponent / denominator if need else np.divide(exponent, denominator, out=np.zeros_like(exponent), where=denominator != 0)
    else: probability = scores
    if dropout_multiplier is not None: probability = probability * dropout_multiplier
    y = np.einsum('nhls,nhsd->nhld', probability, v).transpose(2, 0, 1, 3).reshape(length, batch, embed)
    y = y @ state['out_proj.weight'].T + state.get('out_proj.bias', 0)
    if unbatched: y = y[:, 0]
    elif layout == 'batch': y = y.swapaxes(0, 1)
    w = probability.mean(1) if average else probability
    if unbatched: w = w[0]
    return (y, w) if need else (y,)


def fixture(layout='sequence', separate=False, bias=True, extra='none', dtype=np.float64):
    rng = np.random.default_rng(5201)
    options = dict(batch_first=layout == 'batch', bias=bias, add_bias_kv=extra == 'bias',
                   add_zero_attn=extra == 'zero', dtype=dtype)
    if separate: options.update(kdim=3, vdim=5)
    m = torch.nn.MultiheadAttention(4, 2, **options)
    with torch.no_grad():
        for p in m.parameters(): p.copy_(torch.tensor(rng.normal(0, .3, p.shape).astype(dtype)))
    dims = [(3, 4), (4, 3 if separate else 4), (4, 5 if separate else 4)]
    def shape(length, width):
        return (length, width) if layout == 'unbatched' else (2, length, width) if layout == 'batch' else (length, 2, width)
    arrays = [np.asfortranarray(rng.normal(0, .4, shape(*d)).astype(dtype)) for d in dims]
    inputs = [torch.tensor(a, requires_grad=True) for a in arrays]
    state = {k: p.detach().numpy().copy() for k, p in m.named_parameters()}
    return rng, m, inputs, arrays, state


@pytest.mark.parametrize('layout', ['sequence', 'batch', 'unbatched'])
@pytest.mark.parametrize('separate,extra', list(itertools.product([False, True], ['none', 'bias', 'zero'])))
@pytest.mark.parametrize('need,average', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('mask_kind', ['none', 'boolean', 'float'])
def test_numpy_values_vjp_directions_and_jvp(layout, separate, extra, need, average, mask_kind):
    rng, m, inputs, arrays, state = fixture(layout, separate, extra=extra)
    batch = 1 if layout == 'unbatched' else 2
    mask, padding = None, None
    if mask_kind == 'boolean':
        mask = rng.random((batch * 2, 3, 4)) < .2; mask[..., 0] = False
        padding = np.zeros((batch, 4), dtype=bool); padding[..., -1] = True
    elif mask_kind == 'float':
        mask = rng.normal(0, .1, (3, 4)); padding = rng.normal(0, .1, (batch, 4))
    if padding is not None and layout == 'unbatched': padding = padding[0]
    tm = None if mask is None else torch.tensor(mask, requires_grad=mask_kind == 'float')
    tp = None if padding is None else torch.tensor(padding, requires_grad=mask_kind == 'float')
    options = dict(layout=layout, bias_tokens=extra == 'bias', zero_token=extra == 'zero', need=need, average=average)
    def call(*x):
        y, w = m(*x, key_padding_mask=tp, attn_mask=tm, need_weights=need, average_attn_weights=average)
        return (y, w) if need else (y,)
    expected = reference(arrays, state, mask=mask, padding=padding, **options)
    outputs = call(*inputs)
    for actual, want in zip(outputs, expected): np.testing.assert_allclose(actual.numpy(), want, atol=2e-14, rtol=2e-13)
    seeds = [rng.normal(size=o.shape) for o in outputs]
    active = inputs + list(m.parameters()) + ([tm, tp] if mask_kind == 'float' else [])
    gradients = torch.autograd.grad(sum((o * torch.tensor(s)).sum() for o, s in zip(outputs, seeds)), active)
    directions = [rng.normal(size=x.shape) for x in active]
    def scalar(xs, ps, mk, pd):
        return sum((o*s).sum() for o,s in zip(reference(xs, ps, mask=mk, padding=pd, **options), seeds))
    eps = 1e-5
    # Vary each input/parameter/mask independently; catches disconnected edges.
    for i, (g, d) in enumerate(zip(gradients, directions)):
        values = []
        for sign in [1, -1]:
            xs = list(arrays); ps = dict(state); mk, pd = mask, padding
            if i < 3: xs[i] = xs[i] + sign * eps * d
            elif i < 3 + len(state):
                name = list(state)[i - 3]; ps[name] = ps[name] + sign * eps * d
            elif i == 3 + len(state): mk = mk + sign * eps * d
            else: pd = pd + sign * eps * d
            values.append(scalar(xs, ps, mk, pd))
        np.testing.assert_allclose((g.numpy()*d).sum(), (values[0]-values[1])/(2*eps), atol=4e-9, rtol=4e-7)
    primal, tangent = torch.func.jvp(call, tuple(inputs), tuple(torch.tensor(d) for d in directions[:3]))
    plus, minus = [reference([x+sign*eps*d for x,d in zip(arrays,directions)], state, mask=mask, padding=padding, **options) for sign in [1,-1]]
    for p, t, want, high, low in zip(primal, tangent, expected, plus, minus):
        np.testing.assert_allclose(p.numpy(), want, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(t.numpy(), (high-low)/(2*eps), atol=4e-9, rtol=4e-7)


@pytest.mark.parametrize('shared', ['self', 'kv', 'distinct'])
@pytest.mark.parametrize('layout', ['sequence', 'batch', 'unbatched'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_packed_identity_optimization_keeps_values_and_shared_gradients(shared, layout, dtype):
    rng, m, xs, arrays, state = fixture(layout, dtype=dtype)
    q, k, v = (xs[0],)*3 if shared == 'self' else (xs[0], xs[1], xs[1]) if shared == 'kv' else xs
    actual = m(q, k, v)[0]
    expected = reference([x.numpy() for x in (q,k,v)], state, layout=layout)[0]
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=3e-6)
    actual.square().sum().backward()
    assert all(x.grad is not None for x in {q,k,v})
    duplicates = [torch.tensor(x.numpy(), requires_grad=True) for x in (q,k,v)]
    m(*duplicates)[0].square().sum().backward()
    for x in {q,k,v}:
        want = sum(d.grad.numpy() for original,d in zip((q,k,v),duplicates) if original is x)
        np.testing.assert_allclose(x.grad.numpy(), want, atol=2e-6, rtol=3e-6)


@pytest.mark.parametrize('need', [True, False])
@pytest.mark.parametrize('source', [0, 1, 5])
def test_functional_static_kv_and_empty_source(need, source):
    rng, m, inputs, arrays, state = fixture()
    static = [rng.normal(size=(4, source, 2)) for _ in range(2)]
    sk, sv = [torch.tensor(x, requires_grad=True) for x in static]
    padding = np.zeros((2, source), dtype=bool)
    y, w = torch.nn.functional.multi_head_attention_forward(*inputs, 4, 2, m.in_proj_weight,
        m.in_proj_bias, None, None, False, 0., m.out_proj.weight, m.out_proj.bias,
        static_k=sk, static_v=sv, key_padding_mask=torch.tensor(padding), need_weights=need)
    expected = reference(arrays, state, static=static, need=need, padding=padding)
    for got, want in zip((y,w), expected): np.testing.assert_allclose(got.numpy(), want, rtol=1e-12, atol=1e-13)
    gq, gk, gv, gs, gt = torch.autograd.grad(y.sum(), [*inputs,sk,sv], allow_unused=True)
    assert gq is not None and gs is not None and gt is not None
    assert gk is None and gv is None
    if not need: assert w is None


@pytest.mark.parametrize('need', [True, False])
def test_fully_masked_math_path_and_nonzero_output_bias(need):
    _, m, xs, _, _ = fixture()
    with torch.no_grad(): m.out_proj.bias.fill_(.7)
    y, w = m(*xs, key_padding_mask=torch.ones(2,4,dtype=torch.bool), need_weights=need)
    if need:
        assert np.isnan(y.numpy()).all() and np.isnan(w.numpy()).all()
    else:
        np.testing.assert_allclose(y.numpy(), .7, atol=0, rtol=0)
        grads = torch.autograd.grad(y.sum(), [*xs, *m.parameters()])
        for g in grads[:-1]: np.testing.assert_array_equal(g.numpy(), 0)
        np.testing.assert_array_equal(grads[-1].numpy(), 6)


@pytest.mark.parametrize('need', [True, False])
@pytest.mark.parametrize('p', [0., .3, 1.])
def test_dropout_replay_backward_does_not_draw_and_eval_disables_dropout(need, p):
    _, m, xs, arrays, state = fixture(); m.dropout = p
    torch.manual_seed(5202); before = np.random.get_state()
    y, w = m(*xs, need_weights=need)
    after = np.random.get_state()
    y.square().sum().backward()
    np.testing.assert_array_equal(np.random.get_state()[1], after[1])
    assert np.random.get_state()[2:] == after[2:]
    np.random.set_state(before); ry, rw = m(*xs, need_weights=need)
    np.testing.assert_array_equal(y.numpy(), ry.numpy())
    if need: np.testing.assert_array_equal(w.numpy(), rw.numpy())
    if p == 1:
        np.testing.assert_allclose(y.numpy(), np.broadcast_to(state['out_proj.bias'], y.shape))
        if need: np.testing.assert_array_equal(w.numpy(), 0)
    m.eval(); before = np.random.get_state()
    got = m(*xs, need_weights=need)
    expected = reference(arrays, state, need=need)
    for a,b in zip(got,expected): np.testing.assert_allclose(a.numpy(), b, atol=1e-13, rtol=1e-12)
    np.testing.assert_array_equal(np.random.get_state()[1], before[1])
    assert np.random.get_state()[2:] == before[2:]


def test_state_hooks_pickle_conversion_optimizer_and_namespace():
    _, m, xs, _, _ = fixture(extra='bias')
    assert list(m.state_dict()) == ['in_proj_weight','in_proj_bias','bias_k','bias_v','out_proj.weight','out_proj.bias']
    assert {k:p._version for k,p in torch.nn.MultiheadAttention(4,2,add_bias_kv=True).named_parameters()} == dict(in_proj_weight=1,in_proj_bias=1,bias_k=1,bias_v=1,**{'out_proj.weight':1,'out_proj.bias':2})
    for clone in [copy.deepcopy(m), pickle.loads(pickle.dumps(m))]:
        np.testing.assert_array_equal(clone(*xs)[0].numpy(), m(*xs)[0].numpy())
        assert all(a is not b for a,b in zip(m.parameters(),clone.parameters()))
    converted = copy.deepcopy(m).float()
    assert all(p.dtype == np.float32 for p in converted.parameters())
    clone = torch.nn.MultiheadAttention(4,2,add_bias_kv=True,dtype=torch.float64)
    clone.load_state_dict(m.state_dict())
    events = []
    handle = clone.register_forward_hook(lambda module,args,result: events.append(result[0].shape))
    optimizer = torch.optim.SGD(clone.parameters(), lr=.01)
    initial = clone.out_proj.weight.detach().numpy().copy()
    clone(*xs)[0].square().sum().backward(); optimizer.step(); handle.remove()
    assert events == [(3,2,4)] and not np.array_equal(initial, clone.out_proj.weight.numpy())
    with as_torch():
        from torch.nn.modules.activation import MultiheadAttention
        from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
        from torch.nn.functional import multi_head_attention_forward
        assert MultiheadAttention is torch.nn.MultiheadAttention
        assert isinstance(m.out_proj, NonDynamicallyQuantizableLinear)
        assert callable(multi_head_attention_forward)
    assert not hasattr(torch.nn, 'NonDynamicallyQuantizableLinear')


@pytest.mark.parametrize('need', [True, False])
def test_causal_hint_requires_mask_and_padding_preserves_explicit_mask(need):
    _, m, xs, arrays, state = fixture()
    with pytest.raises(RuntimeError, match='attn_mask'): m(*xs, is_causal=True, need_weights=need)
    mask = np.triu(np.ones((3,4),dtype=bool),1)
    y,w = m(*xs, attn_mask=torch.tensor(mask), is_causal=True, need_weights=need)
    expected = reference(arrays,state,mask=mask,need=need)
    for a,b in zip((y,w),expected): np.testing.assert_allclose(a.numpy(),b,atol=1e-13,rtol=1e-12)
    padding = np.zeros((2,4),dtype=bool)
    arbitrary = mask.copy(); arbitrary[:, -1] = False
    got = m(*xs, attn_mask=torch.tensor(arbitrary), key_padding_mask=torch.tensor(padding), is_causal=True, need_weights=need)
    want = reference(arrays,state,mask=arbitrary,padding=padding,need=need)
    for a,b in zip(got,want): np.testing.assert_allclose(a.numpy(),b,atol=1e-13,rtol=1e-12)


def test_mask_dtype_validation_merge_masks_and_warnings():
    _,m,xs,_,_ = fixture()
    with pytest.raises(RuntimeError, match='dtype'): m(*xs,attn_mask=torch.zeros(3,4,dtype=torch.float32))
    assert m(*xs,attn_mask=torch.zeros(3,4,dtype=torch.float32),need_weights=False)[1] is None
    with pytest.warns(UserWarning,match='mismatched'):
        m(*xs,attn_mask=torch.zeros(3,4,dtype=torch.float64),key_padding_mask=torch.zeros(2,4,dtype=torch.bool))
    with pytest.raises(AssertionError,match='boolean'): m(*xs,attn_mask=torch.zeros(3,4,dtype=torch.int64))
    mask = torch.tensor([[False,True,False],[False,False,True],[True,False,False]])
    padding = torch.tensor([[False,False,True],[True,False,False]])
    merged,kind = m.merge_masks(mask,padding,torch.ones(2,3,4))
    assert kind == 2
    np.testing.assert_array_equal(merged.numpy(),np.broadcast_to(mask.numpy()[None,None] | padding.numpy()[:,None,None],(2,2,3,3)))
    assert m.merge_masks(None,padding,None) == (padding,1)
    assert m.merge_masks(None,None,None) == (None,None)


@pytest.mark.parametrize('options,error', [
    ({'embed_dim':0},ValueError), ({'num_heads':0},ValueError),
    ({'num_heads':3},AssertionError), ({'num_heads':True},TypeError),
    ({'dropout':-.1},ValueError), ({'dropout':float('nan')},ValueError),
    ({'batch_first':1},TypeError), ({'kdim':0},ValueError),
])
def test_constructor_guards(options,error):
    with pytest.raises(error): torch.nn.MultiheadAttention(**dict({'embed_dim':4,'num_heads':2},**options))


@pytest.mark.parametrize('need', [True, False])
@pytest.mark.parametrize('separate', [True, False])
@pytest.mark.parametrize('masked', [True, False])
def test_reverse_of_jvp_reaches_inputs_parameters_and_directions(need,separate,masked):
    rng,m,xs,arrays,state = fixture('batch',separate,extra='bias')
    mask = np.array([[False,True,False,False],[False,False,True,False],[False,False,False,True]]) if masked else None
    dirs = [rng.normal(size=x.shape) for x in xs]
    directions = [torch.tensor(d,requires_grad=True) for d in dirs]
    def call(*x):
        y,w=m(*x,attn_mask=None if mask is None else torch.tensor(mask),need_weights=need,average_attn_weights=False)
        return (y,w) if need else (y,)
    primal,tangent = torch.func.jvp(call,tuple(xs),tuple(directions))
    seeds = [rng.normal(size=t.shape) for t in tangent]
    active = xs+list(m.parameters())+directions
    gradients = torch.autograd.grad(sum((t*torch.tensor(s)).sum() for t,s in zip(tangent,seeds)),active,allow_unused=True)
    names=['q','k','v',*dict(m.named_parameters()),'dq','dk','dv']
    assert [name for name,g in zip(names,gradients) if g is None] == ['out_proj.bias']
    options = dict(layout='batch',bias_tokens=True,mask=mask,need=need,average=False)
    def tangent_loss(inputs,params,ds):
        h=2e-5
        hi,lo=[reference([x+sign*h*d for x,d in zip(inputs,ds)],params,**options) for sign in [1,-1]]
        return sum(((a-b)*s).sum() for a,b,s in zip(hi,lo,seeds))/(2*h)
    h=2e-4
    for i,g in enumerate(gradients):
        d=rng.normal(size=active[i].shape); scalars=[]
        for sign in [1,-1]:
            inputs=list(arrays);params=dict(state);ds=list(dirs)
            if i<3:inputs[i]=inputs[i]+sign*h*d
            elif i<3+len(state):
                name=list(state)[i-3];params[name]=params[name]+sign*h*d
            else:
                j=i-3-len(state);ds[j]=ds[j]+sign*h*d
            scalars.append(tangent_loss(inputs,params,ds))
        np.testing.assert_allclose(0 if g is None else (g.numpy()*d).sum(),(scalars[0]-scalars[1])/(2*h),atol=5e-7,rtol=5e-5)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_encoder_training_checkpoint_and_stochastic_continuation(dtype):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();opts=dict(dtype=dtype)
            self.embed=torch.nn.Embedding(8,4,**opts)
            self.norm=torch.nn.LayerNorm(4,**opts)
            self.attn=torch.nn.MultiheadAttention(4,2,batch_first=True,dropout=.25,**opts)
            self.ff=torch.nn.Sequential(torch.nn.Linear(4,8,**opts),torch.nn.GELU(),torch.nn.Linear(8,4,**opts))
            self.head=torch.nn.Linear(4,3,**opts)
        def forward(self,ids):
            x=self.embed(ids);q=self.norm(x)
            x=x+self.attn(q,q,q,need_weights=False)[0]
            return self.head((x+self.ff(x)).mean(1))
    def create():
        m=Encoder();o=torch.optim.AdamW(m.parameters(),lr=.004,amsgrad=True)
        return m,o,torch.optim.lr_scheduler.StepLR(o,2,.8)
    def step(m,o,s):
        o.zero_grad();torch.nn.CrossEntropyLoss()(m(ids),labels).backward();o.step();s.step()
    torch.manual_seed(5203);m,o,s=create()
    ids=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.long);labels=torch.tensor([0,2],dtype=torch.long)
    for _ in range(3):step(m,o,s)
    state=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict()))
    with torch.random.fork_rng():second,so,ss=create()
    second.load_state_dict(state['model']);ss.load_state_dict(state['scheduler']);so.load_state_dict(state['optimizer'])
    for _ in range(4):
        with torch.random.fork_rng():step(m,o,s)
        step(second,so,ss)
        for a,b in zip(m.parameters(),second.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())
    m.eval()
    with torch.no_grad():
        first=m(ids);again=m(ids)
        assert not first.requires_grad
        np.testing.assert_array_equal(first.numpy(),again.numpy())


@pytest.mark.parametrize('forward',[False,True])
def test_saved_versions_reject_mutation_before_backward(forward):
    _,m,xs,_,_=fixture()
    if forward:
        p,t=torch.func.jvp(lambda *x:m(*x,need_weights=False)[0],tuple(xs),tuple(torch.ones_like(x) for x in xs))
        loss=p.square().sum()+t.square().sum()
    else:loss=m(*xs,need_weights=False)[0].square().sum()
    with torch.no_grad():m.in_proj_weight.add_(.01)
    with pytest.raises(RuntimeError,match='modified in-place'):loss.backward()
