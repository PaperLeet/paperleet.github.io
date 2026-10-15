import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.retrieval.lightgcn.toy import scalar_graph, scalar_propagation
from problems.retrieval.lightgcn.experiment import run


@pytest.mark.t0
def test_graph_contract_precision_empty_axes_strides_and_validation(impl):
    """Boolean histories support empty axes and strided layouts without input mutation."""
    for users,items in ((3,4),(0,4),(3,0),(0,0)):
        history = torch.zeros(users,2*items,dtype=torch.bool)[:,::2]
        before = history.clone()
        for dtype in (None,torch.float64):
            graph = impl.normalized_bipartite_graph(history,dtype=dtype)
            assert graph.shape==(users+items,users+items) and graph.device==history.device
            assert graph.dtype==(torch.float32 if dtype is None else dtype)
            close(graph,torch.zeros(users+items,users+items,dtype=graph.dtype))
        close(history,before)
    for bad in (torch.zeros(2,3),torch.zeros(2,3,dtype=torch.long),torch.zeros(2,3,1,dtype=torch.bool)):
        with pytest.raises(ValueError):impl.normalized_bipartite_graph(bad)
    with pytest.raises(ValueError):impl.normalized_bipartite_graph(torch.zeros(2,3,dtype=torch.bool),dtype=torch.long)


@pytest.mark.t0
def test_propagation_contract_depth_precision_strides_and_supplied_validation(impl):
    """Layer zero is identity; both precisions and zero nodes are supported without mutation."""
    for dtype in (torch.float32,torch.float64):
        for nodes in (0,4):
            initial = torch.randn(nodes,6,dtype=dtype)[:,::2]
            graph = torch.zeros(nodes,2*nodes,dtype=dtype)[:,::2]
            old_initial,old_graph = initial.clone(),graph.clone()
            result = impl.lightgcn_embeddings(initial,graph,0)
            assert result.shape==initial.shape and result.dtype==dtype and result.device==initial.device
            close(result,initial);close(initial,old_initial);close(graph,old_graph)
    initial,graph=torch.zeros(3,2),torch.zeros(3,3)
    for depth in (-1,True,1.5):
        with pytest.raises(ValueError):impl.lightgcn_embeddings(initial,graph,depth)
    for x,g in ((torch.zeros(3,0),graph),(initial,torch.zeros(2,2)),(initial,graph.double()),(initial.long(),graph.long()),
                (initial+math.inf,graph),(initial,graph+math.nan)):
        with pytest.raises(ValueError):impl.lightgcn_embeddings(x,g)


@pytest.mark.t1
def test_hand_computed_symmetric_degrees_and_zero_layer_average(impl):
    """One-hop hand values distinguish endpoint-degree normalization and inclusion of E0."""
    history=torch.tensor([[True,True,False],[False,True,False]])
    h=1/math.sqrt(2)
    expected=torch.tensor([[0,0,h,.5,0],[0,0,0,h,0],[h,0,0,0,0],[.5,h,0,0,0],[0,0,0,0,0]],dtype=torch.float64)
    graph=impl.normalized_bipartite_graph(history,dtype=torch.float64)
    close(graph,expected,atol=1e-12,rtol=1e-12)
    initial=torch.tensor([[float(n),-float(n)] for n in range(1,6)],dtype=graph.dtype)
    wanted=[(3+3*h)/2,(2+4*h)/2,(3+h)/2,(4+.5+2*h)/2,2.5]
    close(impl.lightgcn_embeddings(initial,graph,1),torch.tensor([[v,-v] for v in wanted],dtype=graph.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_graph_across_unbalanced_degrees_and_isolates(impl):
    """Independent endpoint counts check every directed edge and zero entry in small graphs."""
    for users,items in ((4,5),(1,3),(0,2),(2,0)):
        history=torch.rand(users,2*items)[:,::2]>.45
        for dtype in (torch.float32,torch.float64):
            graph=impl.normalized_bipartite_graph(history,dtype=dtype)
            expected=torch.tensor(scalar_graph(history.tolist(),items),dtype=dtype).reshape(users+items,users+items)
            tolerance=3e-6 if dtype==torch.float32 else 1e-12
            close(graph,expected,atol=tolerance,rtol=tolerance)


@pytest.mark.t2
def test_independent_scalar_multihop_propagation_for_multiple_depths(impl):
    """Nested scalar loops independently verify all nodes and features through depth four."""
    for dtype in (torch.float32,torch.float64):
        graph=torch.randn(5,5,dtype=dtype)*.2
        initial=torch.randn(5,6,dtype=dtype)[:,::2]
        for layers in (0,1,2,4):
            expected=scalar_propagation(initial.tolist(),graph.tolist(),layers)
            tolerance=3e-6 if dtype==torch.float32 else 1e-12
            close(impl.lightgcn_embeddings(initial,graph,layers),torch.tensor(expected,dtype=dtype),atol=tolerance,rtol=tolerance)


@pytest.mark.t2
def test_two_hop_collaborative_influence_uses_shared_item_degree(impl):
    """A shared item transmits a hand-computed two-hop signal; disconnected users receive zero."""
    history=torch.tensor([[True,True,False],[False,True,False],[False,False,True]])
    graph=impl.normalized_bipartite_graph(history,dtype=torch.float64)
    initial=torch.zeros(6,1,dtype=graph.dtype);initial[1,0]=1.
    result=impl.lightgcn_embeddings(initial,graph,2)
    close(result[0,0],torch.tensor(1/(6*math.sqrt(2)),dtype=graph.dtype),atol=1e-12,rtol=1e-12)
    close(result[2,0],torch.tensor(0.,dtype=graph.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_embedding_and_operator_gradients_against_scalar_finite_differences(impl):
    """Both differentiable inputs retain every propagation path, including non-initial layers."""
    initial=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    graph=(torch.randn(4,4,dtype=torch.float64)*.2).requires_grad_()
    weights=torch.tensor([[.2,-.4],[.7,.3],[-.1,.9],[.8,-.6]],dtype=initial.dtype)
    actual=(impl.lightgcn_embeddings(initial,graph,3)*weights).sum()
    actual.backward()
    assert initial.grad is not None and graph.grad is not None,'Both differentiable inputs need gradients'
    x,g=initial.detach().tolist(),graph.detach().tolist()
    def objective():
        out=scalar_propagation(x,g,3)
        return sum(a*b for row,w in zip(out,weights.tolist()) for a,b in zip(row,w))
    for values,gradient in ((x,initial.grad),(g,graph.grad)):
        expected=torch.zeros_like(gradient)
        for r,row in enumerate(values):
            for c,original in enumerate(row):
                row[c]=original+1e-6;above=objective()
                row[c]=original-1e-6;below=objective()
                row[c]=original;expected[r,c]=(above-below)/2e-6
        close(gradient,expected,atol=2e-8,rtol=2e-6)


@pytest.mark.t2
def test_linear_superposition_isolated_nodes_and_input_protection(impl):
    """No nonlinearity or implicit self loop: isolated nodes retain exactly E0/(K+1)."""
    history=torch.tensor([[True,False,False],[False,False,False]])
    graph=impl.normalized_bipartite_graph(history,dtype=torch.float64)
    x,y=torch.randn(5,2,dtype=graph.dtype),torch.randn(5,2,dtype=graph.dtype)
    old_x,old_g=x.clone(),graph.clone()
    for layers in (1,2,4):
        result=impl.lightgcn_embeddings(x,graph,layers)
        close(impl.lightgcn_embeddings(2*x-3*y,graph,layers),2*result-3*impl.lightgcn_embeddings(y,graph,layers),atol=1e-12,rtol=1e-12)
        close(result[torch.tensor([1,3,4])],x[torch.tensor([1,3,4])]/(layers+1),atol=1e-12,rtol=1e-12)
    close(x,old_x);close(graph,old_g)


@pytest.mark.t2
def test_user_and_item_relabeling_commutes_with_graph_and_propagation(impl):
    """Separate user/item permutations preserve corresponding graph entries and final embeddings."""
    history=torch.tensor([[True,True,False],[False,True,False],[False,False,True]])
    u,i=torch.tensor([2,0,1]),torch.tensor([1,2,0])
    permutation=torch.cat([u,3+i])
    graph=impl.normalized_bipartite_graph(history,dtype=torch.float64)
    relabeled=impl.normalized_bipartite_graph(history[u][:,i],dtype=graph.dtype)
    close(relabeled,graph[permutation][:,permutation],atol=1e-12,rtol=1e-12)
    initial=torch.randn(6,3,dtype=graph.dtype)
    close(impl.lightgcn_embeddings(initial[permutation],relabeled,3),impl.lightgcn_embeddings(initial,graph,3)[permutation],atol=1e-12,rtol=1e-12)


@pytest.mark.t3
def test_actual_initial_embedding_training_learns_overlapping_preferences(impl,artifact):
    """160 actual updates learn 24 training preferences using only 30 initial-table parameters."""
    result=run(impl.normalized_bipartite_graph,impl.lightgcn_embeddings)
    artifact('lightgcn_training',result);m=result['metrics']
    assert m['updates']==160 and m['trainable_parameters']==30 and m['triples']==24 and m['layers']==2
    assert m['item_only_accuracy_upper_bound']==17/24
    assert m['maximum_graph_error']<1e-12 and m['maximum_isolated_error']<1e-12 and m['maximum_scalar_error']<1e-12
    assert m['final_pair_accuracy']==1 and m['final_training_recall_at_2']==1
    assert m['final_pair_accuracy']>m['initial_pair_accuracy'] and m['final_pair_accuracy']>m['item_only_accuracy_upper_bound']
    assert m['final_loss']<.3*m['initial_loss'] and m['minimum_final_margin']>1.5
