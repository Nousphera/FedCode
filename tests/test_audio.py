import numpy as np
import torch
from datasets.spcm import AudioTransform
from models.yamnet import YAMNet


def test_audio_transform_and_real_yamnet_update():
    torch.manual_seed(17)
    transform = AudioTransform(length=16000)
    x = torch.stack(
        [transform(torch.randn(1, 12000)), transform(torch.randn(1, 20000))]
    )
    assert x.shape == (2, 1, 64, 101)
    assert torch.isfinite(x).all()
    model = YAMNet(num_classes=12)
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 11]))
    loss.backward()
    optimizer.step()
    assert np.isfinite(loss.item())
    assert not torch.equal(before, model.classifier.weight)
