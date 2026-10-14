"""Deterministic RGB images for method/transport smoke tests only."""
import numpy as np
import torch


def get_dataset(split_fn=None,id=0,num_shards=1,return_eval_ds=False,batch_size=8,seed=42):
    rng=np.random.default_rng(seed+10000 if return_eval_ds else seed)
    count=8 if return_eval_ds else 24
    images=rng.normal(size=(count,3,32,32)).astype(np.float32)
    labels=np.arange(count)%2
    if not return_eval_ds:
        ids=np.array_split(np.random.default_rng(seed).permutation(count),num_shards)[id]
        if not len(ids): raise ValueError('Synthetic fixture requires at least one example per client')
        images,labels=images[ids],labels[ids]
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(images),torch.from_numpy(labels).long())
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=not return_eval_ds,
        generator=torch.Generator().manual_seed(seed))
    return loader,2,len(dataset)
