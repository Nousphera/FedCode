from collections import OrderedDict
import numpy as np
import pytest
import torch


def test_resnet_factory_instantiates_paper_resnet20():
    from network import get_model_fn
    model=get_model_fn('resnet20')(num_classes=3)
    assert model(torch.zeros(2,3,32,32)).shape==(2,3)


def test_codebook_reconstruction_preserves_buffers_and_dtypes():
    from utils.fed_utils import model_state_from_aggregated_clusters
    state=OrderedDict(weight=np.array([-1.,.1,2.],np.float32),counter=np.array(7,np.int64))
    actual=model_state_from_aggregated_clusters(state,np.array([-.5,.5]))
    np.testing.assert_array_equal(actual['weight'],[-.5,.5,.5])
    assert actual['weight'].dtype==np.float32
    assert actual['counter']==7 and actual['counter'].dtype==np.int64


def test_bitpacked_calibration_reconstructs_exact_cluster_assignments():
    from utils.codec import pack_calibration,unpack_calibration
    state=OrderedDict(weight=np.zeros((2,3),np.float32),count=np.array(7,np.int64))
    centers=np.array([-.5,0,.5],np.float32)
    labels=np.array([0,1,2,2,1,0])
    payload=pack_calibration(state,centers,labels)
    out=unpack_calibration(payload)
    np.testing.assert_array_equal(out['weight'],centers[labels].reshape(2,3))
    assert out['count']==7 and out['count'].dtype==np.int64
    # 6 two-bit indices occupy two bytes, independent of model metadata.
    assert payload[2].nbytes==2


def test_cluster_aggregation_concatenates_instead_of_averaging_centers():
    from utils.fed_utils import _FedAvg
    state=OrderedDict(weight=np.array([-.8,.15,.9],np.float32))
    arrays,_=_FedAvg.aggregate_clusters([([np.array([-1.,.1])],10),([np.array([.3,1.])],30)],state)
    np.testing.assert_allclose(arrays[0],[-1.,.1,1.])


def test_sparse_iid_partitions_keep_singleton_indices():
    from utils.shard import create_iid_shards
    parts=create_iid_shards(np.arange(5),5,5,5,seed=7)
    assert sorted(sum(parts.values(),[]))==list(range(5))


def test_yamnet_shape_is_independent_of_batch_size():
    from models.yamnet import YAMNet
    model=YAMNet(num_classes=12).eval()
    x=torch.randn(2,1,64,101)
    with torch.no_grad():
        together=model(x)
        separate=torch.cat([model(row[None]) for row in x])
    assert together.shape==(2,12)
    torch.testing.assert_close(together,separate,atol=1e-5,rtol=1e-4)


def test_first_participation_during_codebook_round_gets_calibration():
    import flwr as fl
    from types import SimpleNamespace
    from utils.fed_utils import _FedAvg
    from utils.codec import is_calibration
    state=OrderedDict(weight=np.linspace(-1,1,12,dtype=np.float32))
    strategy=_FedAvg(state,2,min_fit_clients=2,min_available_clients=2,
        on_fit_config_fn=lambda rnd:{'broadcast_clusters':True})
    strategy.initialized_clients.add('0')
    manager=SimpleNamespace(num_available=lambda:2,
        sample=lambda **kw:[SimpleNamespace(cid='0'),SimpleNamespace(cid='1')])
    outputs=strategy.configure_fit(3,fl.common.ndarrays_to_parameters(list(state.values())),manager)
    assert outputs[0][1].config['broadcast_clusters']
    assert not outputs[1][1].config['broadcast_clusters']
    assert is_calibration(fl.common.parameters_to_ndarrays(outputs[1][1].parameters))


def test_impossible_dirichlet_partition_fails_without_looping():
    from utils.shard import create_noniid_shards
    with pytest.raises(ValueError,match='10 samples'):
        create_noniid_shards(np.arange(4),2,4,4)
