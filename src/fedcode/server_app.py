"""Native Flower orchestration with codebook aggregation and periodic calibration."""

from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from .client_app import partitions
from .codec import ModelCodec
from .config import defaults, validate, calibration_round
from .data import load_datasets, data_metadata
from .models import make_model
from .training import evaluate, load_state, numpy_state, seed_everything

app = ServerApp()


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(set(nodes)) != len(nodes) or len(nodes) > count:
            raise ValueError('Unexpected Flower SuperNodes')
        if len(nodes) == count:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Only {len(nodes)}/{count} Flower SuperNodes registered')
        time.sleep(.25)


def replies(grid, messages, timeout):
    expected = {m.metadata.dst_node_id: m for m in messages}
    seen = set()
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if reply.has_error() or node not in expected or node in seen:
            raise RuntimeError('Each selected client must return one successful update')
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError('Reply round mismatch')
        seen.add(node)
        yield reply
    if seen != set(expected):
        raise RuntimeError('Missing client replies; round was not aggregated')


def run_server(grid, cfg):
    cfg = validate(defaults() | cfg)
    seed_everything(cfg['seed'], cfg['torch-threads'])
    data, test = load_datasets(cfg['dataset'], cfg['data-root'])
    shards = partitions(data, cfg)
    model = make_model(cfg['model'], len(data.classes))
    state = numpy_state(model)
    codec = ModelCodec(state, [n for n, p in model.named_parameters() if p.requires_grad])
    count = len(shards)
    nodes = wait_for_nodes(grid, count, cfg['registration-timeout'])
    completed = [0] * count

    def message(node, rnd, cid=0, payload=None):
        content = RecordDict({'config': ConfigRecord({'round': rnd, 'phase': 'train' if rnd else 'identify',
                                                     'client-completed': completed[cid]})})
        if payload is not None:
            content['model'] = ArrayRecord(numpy_ndarrays=payload)
        return Message(content, dst_node_id=node, message_type='train', group_id=str(rnd))

    mapping = {}
    for reply in replies(grid, [message(node, 0) for node in nodes], cfg['reply-timeout']):
        cid = reply.content['metrics']['client-id']
        if type(cid) != int or cid in mapping or not 0 <= cid < count:
            raise ValueError('Client partition IDs must be unique and complete')
        mapping[cid] = reply.metadata.src_node_id
    rng = np.random.default_rng(cfg['seed'])
    compressed = cfg['method'] in {'fedcode', 'clustered-fedavg'}
    history = []
    for rnd in range(1, cfg['num-rounds'] + 1):
        selected = sorted(rng.choice(count, max(1, math.floor(count * cfg['participation'])), replace=False).tolist())
        if compressed:
            centers, labels = codec.cluster(state, cfg['num-clusters'], seed=cfg['kmeans-seed'])
            calibration, codebook = codec.calibration(state, centers, labels), codec.codebook(state, centers)
        messages, down_modes = [], {}
        for cid in selected:
            payload = (calibration if calibration_round(rnd, cfg, 'downstream', completed[cid]) else codebook) if compressed else codec.raw(state)
            down_modes[cid] = codec.kind(payload)
            messages.append(message(mapping[cid], rnd, cid, payload))
        down_bytes = sum(m.content['model'].count_bytes() for m in messages)
        down_arrays = sum(codec.payload_bytes(m.content['model'].to_numpy_ndarrays()) for m in messages)
        updates = []
        expected_kind = ('calibration' if calibration_round(rnd, cfg, 'upstream') else 'codebook') if compressed else 'raw'
        for reply in replies(grid, messages, cfg['reply-timeout']):
            metric = dict(reply.content['metrics'])
            if set(metric) != {'client-id', 'num-examples', 'client-completed', 'loss', 'accuracy'}:
                raise ValueError('Unexpected client metrics')
            cid, n = metric['client-id'], metric['num-examples']
            if (type(cid) != int or cid not in selected or mapping[cid] != reply.metadata.src_node_id
                    or type(n) != int or n != len(shards[cid])
                    or type(metric['client-completed']) != int
                    or metric.get('client-completed') != completed[cid] + 1
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in metric.values())
                    or metric['loss'] < 0 or not 0 <= metric['accuracy'] <= 1):
                raise ValueError('Invalid client metrics or sample count')
            payload = reply.content['model'].to_numpy_ndarrays()
            if codec.kind(payload) != expected_kind:
                raise ValueError('Unexpected upstream encoding')
            metric |= {'downstream': down_modes[cid], 'upstream': expected_kind,
                       'model-record-bytes': reply.content['model'].count_bytes(),
                       'payload-bytes': codec.payload_bytes(payload)}
            updates.append((cid, payload, n, metric))
        updates.sort(key=lambda row: row[0])
        payloads, counts = [row[1] for row in updates], [row[2] for row in updates]
        state = (codec.aggregate_codebooks(state, payloads, counts) if expected_kind == 'codebook'
                 else codec.average([codec.apply(payload) for payload in payloads], counts))
        load_state(model, state)
        for cid in selected:
            completed[cid] += 1
        evaluation = evaluate(model, test, cfg)
        history.append({'round': rnd, 'clients': [row[3] for row in updates],
                        'train-loss': sum(row[3]['loss'] * row[2] for row in updates) / sum(counts),
                        'downstream-record-bytes': down_bytes,
                        'upstream-record-bytes': sum(row[3]['model-record-bytes'] for row in updates),
                        'downstream-payload-bytes': down_arrays,
                        'upstream-payload-bytes': sum(row[3]['payload-bytes'] for row in updates), **evaluation})
        print(json.dumps({'round': rnd, 'accuracy': evaluation['accuracy'], 'loss': evaluation['loss']}), flush=True)
    return model, {
        'schema': 1, 'config': cfg, 'classes': data.classes,
        'data-preparation': data_metadata(cfg['dataset'], cfg['data-root']),
        'partitions': [np.asarray(shard).tolist() for shard in shards], 'rounds': history,
        'best-accuracy': max(row['accuracy'] for row in history),
        'communication': {'record-bytes': 'Serialized model arrays and array keys; excludes Flower transport framing.',
                          'payload-bytes': 'Encoded model arrays including schema, codebooks, indices and auxiliary buffers.'},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'flower': version('flwr')},
    }


@app.main()
def main(grid: Grid, context: Context):
    cfg = validate(defaults() | context.run_config)
    path = Path(cfg['output'])
    checkpoint = path.with_suffix('.pt')
    if path.exists() or checkpoint.exists():
        raise FileExistsError('Output exists; choose a new --output path')
    path.parent.mkdir(parents=True, exist_ok=True)
    model, result = run_server(grid, cfg)
    with checkpoint.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'classes': result['classes'], 'config': cfg}, stream)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
