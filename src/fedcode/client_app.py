"""Stateful native Flower client: receive, train, cluster and transmit."""

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .codec import ModelCodec
from .config import defaults, validate, calibration_round
from .data import load_datasets, partition_indices
from .models import make_model
from .training import load_state, numpy_state, seed_everything, train_local

app = ClientApp()


def partitions(data, cfg):
    if cfg['method'] == 'centralized':
        import numpy as np
        return [np.arange(len(data))]
    return partition_indices(data.targets, cfg['num-clients'], alpha=cfg['dirichlet-alpha'],
                             seed=cfg['data-seed'], min_size=2)


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = context.node_config['partition-id']
    if type(cid) != int or not 0 <= cid < (1 if cfg['method'] == 'centralized' else cfg['num-clients']):
        raise ValueError('Invalid client partition')
    incoming = msg.content['config']
    rnd, phase = incoming['round'], incoming['phase']
    if type(rnd) != int or not 0 <= rnd <= cfg['num-rounds'] or phase not in {'identify', 'train'}:
        raise ValueError('Invalid experiment message')
    if (phase == 'identify') != (rnd == 0):
        raise ValueError('Invalid initialization round')
    if phase == 'identify':
        return Message(RecordDict({'metrics': MetricRecord({'client-id': cid})}), reply_to=msg)
    progress = context.state.get('progress', ConfigRecord({'completed': 0, 'last-round': 0}))
    completed = incoming['client-completed']
    if (type(completed) != int or completed != progress['completed']
            or not 0 <= completed < rnd or rnd <= progress['last-round']):
        raise ValueError('Client participation state does not match the server round')
    seed_everything(cfg['seed'], cfg['torch-threads'])
    data, _ = load_datasets(cfg['dataset'], cfg['data-root'])
    indices = partitions(data, cfg)[cid]
    model = make_model(cfg['model'], len(data.classes))
    codec = ModelCodec(numpy_state(model), [n for n, p in model.named_parameters() if p.requires_grad])
    compressed = cfg['method'] in {'fedcode', 'clustered-fedavg'}
    payload = msg.content['model'].to_numpy_ndarrays()
    expected = ('calibration' if calibration_round(rnd, cfg, 'downstream', completed) else 'codebook') if compressed else 'raw'
    if codec.kind(payload) != expected:
        raise ValueError('Unexpected downstream encoding')
    previous = None
    if expected == 'codebook':
        if 'model' not in context.state:
            raise ValueError('Codebook reception requires stored client weights')
        previous = {key: value.numpy() for key, value in context.state['model'].to_torch_state_dict().items()}
    load_state(model, codec.apply(payload, previous))
    optimizer = context.state['optimizer'].to_torch_state_dict() if 'optimizer' in context.state else None
    metrics, optimizer = train_local(model, data, indices, cfg,
                                     (cfg['seed'] + rnd * cfg['num-clients'] + cid) % (2 ** 32), optimizer)
    if cfg['method'] == 'centralized':
        context.state['optimizer'] = ArrayRecord(optimizer)
    state = numpy_state(model)
    if compressed:
        centers, labels = codec.cluster(state, cfg['num-clusters'], seed=cfg['kmeans-seed'])
        calibration = codec.calibration(state, centers, labels)
        # The next codebook must act on this client's latest quantized weight positions.
        load_state(model, codec.apply(calibration))
        context.state['model'] = ArrayRecord(model.state_dict())
        payload = calibration if calibration_round(rnd, cfg, 'upstream') else codec.codebook(state, centers)
    else:
        payload = codec.raw(state)
    context.state['progress'] = ConfigRecord({'completed': completed + 1, 'last-round': rnd})
    return Message(RecordDict({
        'model': ArrayRecord(numpy_ndarrays=payload),
        'metrics': MetricRecord({'client-id': cid, 'num-examples': len(indices),
                                 'client-completed': completed + 1, **metrics}),
    }), reply_to=msg)
