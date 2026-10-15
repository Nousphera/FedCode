"""Local Adam optimization and held-out classification metrics."""

import random
import numpy as np
import torch
from torch.nn import functional as F
from .data import make_loader


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numpy_state(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict({name: torch.from_numpy(np.asarray(value).copy()) for name, value in state.items()}, strict=True)


def train_local(model, data, indices, cfg, seed, optimizer_state=None):
    seed_everything(seed, cfg['torch-threads'])
    model.to(cfg['device']).train()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg['learning-rate'])
    if optimizer_state:
        for key, value in optimizer_state.items():
            index, name = key.split('/', 1)
            parameter = parameters[int(index)]
            optimizer.state[parameter][name] = value.cpu() if name == 'step' else value.to(cfg['device'])
    loss_sum, correct, seen = 0., 0, 0
    for epoch in range(cfg['local-epochs']):
        loader = make_loader(data, cfg['batch-size'], indices=indices, shuffle=True,
                             seed=(seed + epoch) % (2 ** 32), num_workers=cfg['num-workers'])
        for inputs, labels in loader:
            inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite local gradient')
            optimizer.step()
            seen += len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(loss.detach()) * len(labels)
    if seen != len(indices) * cfg['local-epochs']:
        raise ValueError('Local training must visit every assigned sample each epoch')
    optimizer_state = {f'{i}/{name}': value.detach().cpu() for i, p in enumerate(parameters)
                       for name, value in optimizer.state[p].items()}
    model.cpu()
    return {'loss': loss_sum / seen, 'accuracy': correct / seen}, optimizer_state


@torch.inference_mode()
def evaluate(model, data, cfg):
    model.to(cfg['device']).eval()
    classes = len(data.classes)
    confusion = torch.zeros((classes, classes), dtype=torch.int64)
    loss_sum, seen = 0., 0
    loader = make_loader(data, cfg['batch-size'], shuffle=False, num_workers=cfg['num-workers'])
    for inputs, labels in loader:
        logits = model(inputs.to(cfg['device'])).cpu()
        loss = F.cross_entropy(logits, labels, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite evaluation loss')
        predicted = logits.argmax(1)
        confusion += torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)
        loss_sum += float(loss)
        seen += len(labels)
    if not seen:
        raise ValueError('Evaluation split is empty')
    model.cpu()
    return {'loss': loss_sum / seen, 'accuracy': float(confusion.diag().sum()) / seen,
            'confusion-matrix': confusion.tolist()}
