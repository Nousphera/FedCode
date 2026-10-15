"""Speech Commands power-mel features: 64 bands, 101 frames, offset 1e-3.

No logarithm is applied. HTK filter-bank construction follows torchaudio;
its copyright and BSD-2-Clause license notice are in LICENSE.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 16_000


class AudioTransform(nn.Module):
    """Map mono 16 kHz float waveforms [1, T] to [1, 64, 101] features.

    Input samples must already be normalized PCM values at 16 kHz. A one-
    second waveform is obtained by symmetric zero-padding or center-cropping.
    Uses a periodic Hann window, reflected STFT padding and unnormalized
    HTK mel filters.
    The data loader checks WAV sample rates before calling this transform.
    """

    def __init__(self, length: int = SAMPLE_RATE):
        super().__init__()
        if isinstance(length, bool) or not isinstance(length, int) or length <= 200:
            raise ValueError("length must be an integer greater than 200 samples")
        self.length = length
        self.register_buffer("window", torch.hann_window(400, periodic=True))
        mel_min = 2595.0 * math.log10(1.0 + 125.0 / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + 7500.0 / 700.0)
        mel_points = torch.linspace(mel_min, mel_max, 66)
        frequencies = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = torch.linspace(0.0, SAMPLE_RATE // 2, 201)
        spacing = frequencies[1:] - frequencies[:-1]
        slopes = frequencies.unsqueeze(0) - bins.unsqueeze(1)
        rising = (-1.0 * slopes[:, :-2]) / spacing[:-1]
        falling = slopes[:, 2:] / spacing[1:]
        filters = torch.max(torch.zeros(1), torch.min(rising, falling))
        self.register_buffer("mel_filters", filters)

    @staticmethod
    def center_pad_or_trim_audio(waveform: torch.Tensor, length: int) -> torch.Tensor:
        samples = waveform.shape[-1]
        if samples < length:
            padding = length - samples
            left = padding // 2
            return F.pad(waveform, (left, padding - left))
        start = (samples - length) // 2
        return waveform[..., start:start + length]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] == 0:
            raise ValueError("Expected a nonempty mono waveform with shape [1, samples]")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("Waveform must contain finite floating-point PCM samples")
        waveform = self.center_pad_or_trim_audio(waveform, self.length)
        spectrum = torch.stft(
            waveform, n_fft=400, hop_length=160, win_length=400,
            window=self.window, center=True, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        )
        power = spectrum.abs().pow(2.0)
        return (power.transpose(-1, -2) @ self.mel_filters).transpose(-1, -2) + 1e-3
