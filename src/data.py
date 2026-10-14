def get_dataset_fn(name='cifar10'):
    if name in ['cifar10','cifar100']:
        from datasets.cifar import get_cifar10_dataset,get_cifar100_dataset
        return get_cifar10_dataset if name=='cifar10' else get_cifar100_dataset
    if name=='spcm':
        from datasets.spcm import get_spcm_dataset
        return get_spcm_dataset
    if name=='synthetic':
        from datasets.synthetic import get_dataset
        return get_dataset
    raise ValueError(f'Unknown dataset: {name}')
