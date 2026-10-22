"""Functional JVP graph ownership, native boundaries and independent derivative fixtures."""
import numpy as np
import pytest
from torchwasm import as_torch, torch

# Reference file SHA256: 87d2e5e3e4be39157e4180a85bf6b6c2da1e7fce31e3d0c913b5d802b7e9545e
_REFERENCES = [{'case': 'layernorm',
  'original_dtype': 'float32',
  'values': {'x': [[0.21872912347316742, -0.16809353232383728, -0.04255164787173271],
                   [0.10689888894557953, 0.051815759390592575, 0.06662540137767792]],
             'weight': [0.06579125672578812, -0.27461519837379456, 0.03565335273742676],
             'bias': [-0.08802737295627594, -0.1674501895904541, -0.12690512835979462],
             'vx': [[1.0, 0.9939560890197754, 0.9758974313735962],
                    [0.9460423588752747, 0.9047516584396362, 0.8525245189666748]],
             'vweight': [0.5403022766113281, 0.44466152787208557, 0.34364575147628784],
             'vbias': [-0.416146844625473, -0.5134528279304504, -0.6045522689819336]},
  'result': [0.0645686926722787, -0.10283096047802184],
  'gradients': {'x': [[[-0.04644067069900092, -0.09688251918180207, 0.14332318988080298],
                       [-0.2072976885386259, -0.6383156543951726, 0.8456133429337982]],
                      [[0.7816839694730282, 1.6058771249149404, -2.3875610943879684],
                       [5.919348450451543, 13.585905219602262, -19.505253670053797]]],
                'weight': [[0.0031612626055044584, -0.47006981662035924, 0.1779088546109479],
                           [1.769228502042749, 4.93615810213481, 1.4581927547542402]],
                'bias': [[0.0023392073686386017, 0.4570096806510011, -0.5534079802168688],
                         [1.3125226713921991, -4.570985976073425, -2.985355813061496]],
                'vx': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       [[-0.04644067069900092, -0.09688251918180209, 0.14332318988080298],
                        [-0.20729768853862618, -0.6383156543951725, 0.8456133429337983]]],
                'vweight': [[0.0, 0.0, 0.0],
                            [0.0031612626055044584, -0.47006981662035924, 0.1779088546109479]],
                'vbias': [[0.0, 0.0, 0.0],
                          [0.0023392073686386017, 0.4570096806510011, -0.5534079802168688]]}},
 {'case': 'layernorm',
  'original_dtype': 'float64',
  'values': {'x': [[0.21872912682734047, -0.16809353436693633, -0.04255164876151732],
                   [0.10689888866759474, 0.05181576078464593, 0.06662540425788138]],
             'weight': [0.06579125352528468, -0.2746151968275666, 0.035653353483312396],
             'bias': [-0.08802737571147048, -0.1674501923247396, -0.12690513317629182],
             'vx': [[1.0, 0.9939560979566968, 0.9758974493306055],
                    [0.946042343528387, 0.9047516632199634, 0.8525245220595057]],
             'vweight': [0.5403023058681398, 0.4446615167417068, 0.34364574631604705],
             'vbias': [-0.4161468365471424, -0.5134528123039594, -0.6045522710579296]},
  'result': [0.06456869499543573, -0.10283095441015211],
  'gradients': {'x': [[[-0.04644066694046749, -0.09688251201941275, 0.14332317895988014],
                       [-0.2072977281186716, -0.6383156635777575, 0.8456133916964287]],
                      [[0.7816839272035964, 1.6058770638041238, -2.3875609910077196],
                       [5.919349105729821, 13.585904356048202, -19.505253461778018]]],
                'weight': [[0.003161222013889754, -0.4700698253065414, 0.17790884606692325],
                           [1.7692287235729618, 4.936158077331011, 1.4581926747690892]],
                'bias': [[0.0023391772627611995, 0.457009682685801, -0.5534079969344269],
                         [1.3125228490518979, -4.570985863736409, -2.9853557827750272]],
                'vx': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       [[-0.04644066694046749, -0.0968825120194127, 0.14332317895988014],
                        [-0.20729772811867195, -0.6383156635777573, 0.8456133916964288]]],
                'vweight': [[0.0, 0.0, 0.0],
                            [0.003161222013889754, -0.4700698253065414, 0.17790884606692325]],
                'vbias': [[0.0, 0.0, 0.0], [0.0023391772627611995, 0.457009682685801, -0.5534079969344269]]}},
 {'case': 'instancenorm_train',
  'original_dtype': 'float32',
  'values': {'x': [[[0.21872912347316742, -0.16809353232383728, -0.04255164787173271],
                    [0.10689888894557953, 0.051815759390592575, 0.06662540137767792]],
                   [[0.06579125672578812, -0.27461519837379456, 0.03565335273742676],
                    [-0.08802737295627594, -0.1674501895904541, -0.12690512835979462]]],
             'weight': [-0.05436957627534866, 0.11980844289064407],
             'bias': [-0.07425706088542938, -0.21970880031585693],
             'vx': [[[1.0, 0.9939560890197754, 0.9758974313735962],
                     [0.9460423588752747, 0.9047516584396362, 0.8525245189666748]],
                    [[0.7899922132492065, 0.717910647392273, 0.6371511220932007],
                     [0.5486898422241211, 0.4535961151123047, 0.3530193865299225]]],
             'vweight': [0.5403022766113281, 0.44466152787208557],
             'vbias': [-0.416146844625473, -0.5134528279304504]},
  'result': [0.4253837374241862, 2.005256489433823],
  'gradients': {'x': [[[[1.8934162132680134e-05, -1.496858814477989e-05, -3.965573987887314e-06],
                        [0.029978205702993192, -0.021972883955607977, -0.008005321747385611]],
                       [[1.3019974872876617e-05, -2.2863044662515197e-05, 9.843069789650923e-06],
                        [0.010047144548966514, -0.010188747098844932, 0.00014160254987825014]]],
                      [[[-0.00037750234987809383, 0.00029948546668361157, 7.801688319445638e-05],
                        [0.13770405118612214, -0.06642990018698024, -0.0712741509991442]],
                       [[-0.00025186944624327123, 0.00045646507069155317, -0.00020459562444839245],
                        [0.052654050245898414, -0.02788814440886987, -0.024765905837029447]]]],
                'weight': [[-0.6521715660486183, 1.4178997966581437], [6.480998138315027, 5.3058003588837]],
                'bias': [[-0.8910847306251526, -2.6365056037902828],
                         [-4.993762135505676, -6.161433935165405]],
                'vx': [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                       [[[1.8934162132682485e-05, -1.4968588144782242e-05, -3.965573987887314e-06],
                         [0.029978205702992883, -0.021972883955607845, -0.008005321747385611]],
                        [[1.3019974872871916e-05, -2.2863044662507556e-05, 9.843069789650923e-06],
                         [0.010047144548966344, -0.010188747098845008, 0.00014160254987825014]]]],
                'vweight': [[0.0, 0.0], [-0.6521715660486183, 1.4178997966581437]],
                'vbias': [[0.0, 0.0], [-0.8910847306251526, -2.6365056037902828]]}},
 {'case': 'instancenorm_train',
  'original_dtype': 'float64',
  'values': {'x': [[[0.21872912682734047, -0.16809353436693633, -0.04255164876151732],
                    [0.10689888866759474, 0.05181576078464593, 0.06662540425788138]],
                   [[0.06579125352528468, -0.2746151968275666, 0.035653353483312396],
                    [-0.08802737571147048, -0.1674501923247396, -0.12690513317629182]]],
             'weight': [-0.054369577731646405, 0.11980843990900887],
             'bias': [-0.07425706446319202, -0.21970880384535085],
             'vx': [[[1.0, 0.9939560979566968, 0.9758974493306055],
                     [0.946042343528387, 0.9047516632199634, 0.8525245220595057]],
                    [[0.7899922314973651, 0.7179106696109434, 0.6371511441985802],
                     [0.5486898605815875, 0.4535961214255773, 0.3530194012193304]]],
             'vweight': [0.5403023058681398, 0.4446615167417068],
             'vbias': [-0.4161468365471424, -0.5134528123039594]},
  'result': [0.4253837465774039, 2.0052564197745837],
  'gradients': {'x': [[[[1.8934162346012375e-05, -1.4968588282138756e-05, -3.965574063866567e-06],
                        [0.029978207456642737, -0.02197288629554913, -0.008005321161094337]],
                       [[1.3019975749461563e-05, -2.286304654325679e-05, 9.84307079380992e-06],
                        [0.010047144247184782, -0.010188746438463868, 0.00014160219127836996]]],
                      [[[-0.00037750236293235235, 0.000299485475751087, 7.80168871813594e-05],
                        [0.13770409510946444, -0.06642993399728728, -0.0712741611121797]],
                       [[-0.00025186946980747547, 0.00045646511983110085, -0.00020459565002376408],
                        [0.05265404088204538, -0.027888135514118575, -0.024765905367931593]]]],
                'weight': [[-0.652171583518165, 1.4178997603275363], [6.4809984892756365, 5.305800217377369]],
                'bias': [[-0.8910847735583042, -2.63650564614421], [-4.993762038565709, -6.161433747647512]],
                'vx': [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                       [[[1.8934162346017077e-05, -1.4968588282129353e-05, -3.965574063866567e-06],
                         [0.029978207456642172, -0.021972886295548912, -0.008005321161094054]],
                        [[1.3019975749461563e-05, -2.2863046543263843e-05, 9.84307079380992e-06],
                         [0.010047144247185271, -0.01018874643846411, 0.00014160219127836996]]]],
                'vweight': [[0.0, 0.0], [-0.652171583518165, 1.4178997603275363]],
                'vbias': [[0.0, 0.0], [-0.8910847735583042, -2.63650564614421]]}},
 {'case': 'batchnorm_train',
  'original_dtype': 'float32',
  'values': {'x': [[[0.21872912347316742, -0.16809353232383728], [-0.04255164787173271, 0.10689888894557953]],
                   [[0.051815759390592575, 0.06662540137767792],
                    [0.06579125672578812, -0.27461519837379456]]],
             'weight': [0.0, 0.025926828384399414],
             'bias': [0.16829419136047363, 0.18088243901729584],
             'vx': [[[1.0, 0.9939560890197754], [0.9758974313735962, 0.9460423588752747]],
                    [[0.9047516584396362, 0.8525245189666748], [0.7899922132492065, 0.717910647392273]]],
             'vweight': [0.0, 0.0],
             'vbias': [0.0, 0.0]},
  'result': [0.24685314313854406, 1.0690387314184896e-06],
  'gradients': {'x': [[[[0.0, 0.0], [-1.794822624817058e-07, 3.9905658061544524e-06]],
                       [[0.0, 0.0], [2.843558869978168e-06, -6.654642413660318e-06]]],
                      [[[0.0, 0.0], [3.6179626795450197e-06, -4.493677188698047e-06]],
                       [[0.0, 0.0], [-6.845828493719281e-06, 7.721543002868784e-06]]]],
                'weight': [[1.504632769052528e-16, 0.20732013469687544],
                           [-1.88079096131566e-17, 8.246583157574391e-05]],
                'bias': [[1.346353530883789, 1.4470595121383667], [0.0, 4.70197740328915e-18]],
                'vx': [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                       [[[0.0, 0.0], [-1.794822624817058e-07, 3.990565806145049e-06]],
                        [[0.0, 0.0], [2.84355886998287e-06, -6.654642413655616e-06]]]],
                'vweight': [[0.0, 0.0], [1.504632769052528e-16, 0.20732013469687544]],
                'vbias': [[0.0, 0.0], [1.346353530883789, 1.4470595121383667]]}},
 {'case': 'batchnorm_train',
  'original_dtype': 'float64',
  'values': {'x': [[[0.21872912682734047, -0.16809353436693633], [-0.04255164876151732, 0.10689888866759474]],
                   [[0.05181576078464593, 0.06662540425788138], [0.06579125352528468, -0.2746151968275666]]],
             'weight': [0.0, 0.025926828523938975],
             'bias': [0.16829419696157932, 0.1808824378757652],
             'vx': [[[1.0, 0.9939560979566968], [0.9758974493306055, 0.946042343528387]],
                    [[0.9047516632199634, 0.8525245220595057], [0.7899922314973651, 0.7179106696109434]]],
             'vweight': [0.0, 0.0],
             'vbias': [0.0, 0.0]},
  'result': [0.24685314905665887, 1.0690386143610842e-06],
  'gradients': {'x': [[[[0.0, 0.0], [-1.7948227551558717e-07, 3.990565993575272e-06]],
                       [[0.0, 0.0], [2.843558925419183e-06, -6.654642643464762e-06]]],
                      [[[0.0, 0.0], [3.617963005316822e-06, -4.493677182561967e-06]],
                       [[0.0, 0.0], [-6.8458277852888036e-06, 7.721541962555106e-06]]]],
                'weight': [[7.52316384526264e-17, 0.2073201358111141],
                           [-3.197344634236622e-16, 8.246582210237851e-05]],
                'bias': [[1.3463535756926346, 1.4470595030061215], [0.0, -6.465218929522581e-17]],
                'vx': [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                       [[[0.0, 0.0], [-1.7948227551558717e-07, 3.990565993575272e-06]],
                        [[0.0, 0.0], [2.843558925419183e-06, -6.654642643462411e-06]]]],
                'vweight': [[0.0, 0.0], [7.52316384526264e-17, 0.2073201358111141]],
                'vbias': [[0.0, 0.0], [1.3463535756926346, 1.4470595030061215]]}}]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind', ['sin', 'cube', 'linear'])
@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('gx', [False, True])
@pytest.mark.parametrize('gv', [False, True])
@pytest.mark.parametrize('strict', [False, True])
def test_functional_jvp_values_graph_and_original_gradient_ownership(dtype, kind, graph, gx, gv, strict):
    data = np.array([-.4, .2], dtype=dtype)
    direction = np.array([.7, -.3], dtype=dtype)
    x = torch.tensor(data, requires_grad=gx)
    v = torch.tensor(direction, requires_grad=gv)
    x.grad = torch.full_like(x, 17); v.grad = torch.full_like(v, 19)
    buffers = (x.grad, v.grad); hooks = []; calls = []
    for item in [x, v]:
        if item.requires_grad: item.register_hook(lambda gradient: hooks.append(gradient))
    def function(a):
        calls.append((torch.is_grad_enabled(), a is x, a.requires_grad, a.is_leaf))
        return a.sin() if kind == 'sin' else a.pow(3) if kind == 'cube' else 3*a
    constant_strict_graph = graph and strict and kind == 'linear' and not gv
    with torch.no_grad():
        if constant_strict_graph:
            with pytest.raises(RuntimeError):
                torch.autograd.functional.jvp(function, x, v, create_graph=graph, strict=strict)
        else:
            output, product = torch.autograd.functional.jvp(function, x, v, create_graph=graph, strict=strict)
        assert not torch.is_grad_enabled()
    assert calls == [(True, False, True, not (graph and gx))] and not hooks
    assert x.grad is buffers[0] and v.grad is buffers[1]
    if constant_strict_graph: return
    value = np.sin(data) if kind == 'sin' else data**3 if kind == 'cube' else 3*data
    slope = np.cos(data) if kind == 'sin' else 3*data**2 if kind == 'cube' else np.full_like(data, 3)
    curvature = -np.sin(data) if kind == 'sin' else 6*data if kind == 'cube' else np.zeros_like(data)
    np.testing.assert_allclose(output.detach().numpy(), value, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(product.detach().numpy(), slope*direction, rtol=2e-6, atol=2e-7)
    assert output.requires_grad == graph
    assert product.requires_grad == (graph and (kind != 'linear' or gv))
    active = tuple(item for item in [x, v] if item.requires_grad)
    if graph and active:
        primal = torch.autograd.grad(output, active, torch.ones_like(output), retain_graph=True, allow_unused=True)
        tangent = torch.autograd.grad(product, active, torch.ones_like(product), retain_graph=True, allow_unused=True) if product.requires_grad else (None,)*len(active)
        cursor = 0
        if gx:
            np.testing.assert_allclose(primal[cursor].numpy(), slope, rtol=2e-6, atol=2e-7)
            if kind == 'linear': assert tangent[cursor] is None
            else: np.testing.assert_allclose(tangent[cursor].numpy(), curvature*direction, rtol=2e-6, atol=2e-7)
            cursor += 1
        if gv:
            assert primal[cursor] is None
            np.testing.assert_allclose(tangent[cursor].numpy(), slope, rtol=2e-6, atol=2e-7)
    assert x.grad is buffers[0] and v.grad is buffers[1]
    assert x.grad.tolist() == [17., 17.] and v.grad.tolist() == [19., 19.]
    assert x.requires_grad == gx and v.requires_grad == gv


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('kind', ['mixed', 'mixed_view', 'identity', 'detach', 'empty', 'view'])
def test_functional_jvp_mixed_no_grad_empty_outputs_and_error_cleanup(dtype, graph, strict, kind):
    data = np.array([-.4, .2], dtype=dtype); vector = np.array([.7, -.3], dtype=dtype)
    x = torch.tensor(data, requires_grad=True); v = torch.tensor(vector, requires_grad=True)
    def function(a):
        if kind == 'empty': return ()
        if kind == 'identity':
            with torch.no_grad(): return a
        if kind == 'view':
            with torch.no_grad(): return a.view_as(a)
        if kind == 'detach': return a.detach().square()
        with torch.no_grad(): frozen = a.square() if kind == 'mixed' else a.view_as(a)
        return a.sin() + frozen
    if kind == 'empty':
        with pytest.raises(RuntimeError if strict else IndexError):
            torch.autograd.functional.jvp(function, x, v, create_graph=graph, strict=strict)
    elif kind == 'view' or kind == 'detach' and strict:
        with pytest.raises(RuntimeError):
            torch.autograd.functional.jvp(function, x, v, create_graph=graph, strict=strict)
    else:
        _, product = torch.autograd.functional.jvp(function, x, v, create_graph=graph, strict=strict)
        target = vector if kind == 'identity' else np.zeros_like(vector) if kind == 'detach' else np.cos(data)*vector
        np.testing.assert_allclose(product.detach().numpy(), target, rtol=2e-6, atol=2e-7)
    with torch.no_grad():
        _, ordinary = torch.func.jvp(lambda a:a.square(), (x,), (v,))
    np.testing.assert_allclose(ordinary.detach().numpy(), 2*data*vector, rtol=2e-6, atol=2e-7)
    assert x.grad is None and v.grad is None


@pytest.mark.parametrize('fixture', _REFERENCES, ids=lambda item:item['case']+'-'+item['original_dtype'])
@pytest.mark.parametrize('vector_grad', [False, True])
@pytest.mark.parametrize('strict', [False, True])
def test_functional_jvp_norm_graphs_match_independent_full_derivative_fixtures(fixture, vector_grad, strict):
    dtype = getattr(torch, fixture['original_dtype'])
    values = fixture['values']
    args = tuple(torch.tensor(values[name], dtype=dtype, requires_grad=True) for name in ['x', 'weight', 'bias'])
    vectors = tuple(torch.tensor(values[name], dtype=dtype, requires_grad=vector_grad) for name in ['vx', 'vweight', 'vbias'])
    for item in args+vectors: item.grad = torch.full_like(item, 7)
    buffers = tuple(item.grad for item in args+vectors)
    calls = []
    def function(a, w, b):
        calls.append(True)
        if fixture['case'] == 'layernorm': value = torch.nn.functional.layer_norm(a, (3,), w, b)
        elif fixture['case'] == 'instancenorm_train': value = torch.nn.functional.instance_norm(a, weight=w, bias=b)
        else: value = torch.nn.functional.batch_norm(a, None, None, w, b, training=True)
        return value.square().sum()
    output = torch.autograd.functional.jvp(function, args, vectors, create_graph=True, strict=strict)
    assert calls == [True]
    active = args+vectors if vector_grad else args
    names = ['x', 'weight', 'bias'] + (['vx', 'vweight', 'vbias'] if vector_grad else [])
    rtol, atol = (1e-4, 3e-6) if fixture['original_dtype'] == 'float32' else (1e-8, 2e-10)
    for component, value in enumerate(output):
        assert value.requires_grad
        np.testing.assert_allclose(value.detach().numpy(), fixture['result'][component], rtol=rtol, atol=atol)
        gradients = torch.autograd.grad(value, active, retain_graph=True, allow_unused=True)
        for name, gradient in zip(names, gradients):
            target = np.asarray(fixture['gradients'][name][component])
            if component == 0 and name.startswith('v'):
                assert gradient is None
                np.testing.assert_array_equal(target, np.zeros_like(target))
            else:
                assert gradient is not None and gradient.dtype == args[0].dtype and not gradient.requires_grad
                np.testing.assert_allclose(gradient.numpy(), target, rtol=rtol, atol=atol)
    assert all(item.grad is buffer for item, buffer in zip(args+vectors, buffers))


def test_functional_jvp_import_and_batchnorm_captured_state_updates_once():
    model = torch.nn.BatchNorm1d(2, dtype=torch.float64)
    x = torch.tensor(np.arange(8).reshape(2,2,2)*.1, dtype=torch.float64, requires_grad=True)
    vector = torch.ones_like(x); calls = []
    def function(a):
        calls.append(True)
        return model(a).square().sum()
    with as_torch() as alias:
        from torch.autograd.functional import jvp
        assert jvp is alias.autograd.functional.jvp
        value, product = jvp(function, x, vector, create_graph=True)
    assert calls == [True] and model.num_batches_tracked.item() == 1
    torch.autograd.grad(value, x, retain_graph=True)
    torch.autograd.grad(product, x, retain_graph=True)
    assert model.num_batches_tracked.item() == 1 and x.grad is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_generic_relu_keeps_zero_reverse_edge_and_hvp_constant_strictness(dtype):
    x = torch.tensor([-.4,.2], dtype=dtype, requires_grad=True)
    _, tangent = torch.func.jvp(torch.relu, (x,), (torch.ones_like(x),))
    assert tangent.requires_grad
    gradient = torch.autograd.grad(tangent, x, torch.ones_like(tangent))[0]
    assert gradient.tolist() == [0., 0.]
    with pytest.raises(RuntimeError):
        torch.autograd.functional.hvp(lambda a:a.relu().sum(), x, torch.ones_like(x), strict=True)
