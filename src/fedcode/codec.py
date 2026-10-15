"""FedCode's global parameter codebook and validated NumPy transport.

Only named trainable parameters are clustered. Auxiliary state (including
BatchNorm statistics) is transported separately and included in payload sizes.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _nearest(values, centers):
    """Binary search with lower-center tie breaking and no P-by-K matrix."""
    wide_centers = centers.astype(np.float64)
    values = np.asarray(values, dtype=np.float64)
    right = np.searchsorted(wide_centers, values).clip(0, len(centers) - 1)
    left = (right - 1).clip(0, len(centers) - 1)
    # For bracketing finite centers at most one distance can overflow. Its
    # infinity still compares correctly, unlike a rounded midpoint threshold.
    with np.errstate(over="ignore"):
        take_left = np.abs(values - wide_centers[left]) <= np.abs(values - wide_centers[right])
    return np.where(take_left, left, right)


def pack_indices(labels, num_centers):
    """Pack canonical little-endian indices, including the zero-bit K=1 case."""
    num_centers = _integer(num_centers, "num_centers", 1)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional integer array")
    if np.any(labels < 0) or np.any(labels >= num_centers):
        raise ValueError("codebook index out of range")
    width = (num_centers - 1).bit_length()
    if width > 63:
        raise ValueError("too many centers")
    if width == 0:
        return np.empty(0, dtype=np.uint8)
    values = labels.astype(np.uint64)
    bits = np.column_stack(
        [((values >> bit) & 1).astype(np.uint8) for bit in range(width)]
    )
    return np.packbits(bits.reshape(-1), bitorder="little")


def unpack_indices(packed, count, num_centers):
    """Reject truncated, oversized, noncanonical or out-of-range indices."""
    count = _integer(count, "count")
    num_centers = _integer(num_centers, "num_centers", 1)
    packed = np.asarray(packed)
    width = (num_centers - 1).bit_length()
    if width > 63 or packed.dtype != np.uint8 or packed.ndim != 1:
        raise ValueError("invalid packed index representation")
    bit_count = count * width
    if len(packed) != (bit_count + 7) // 8:
        raise ValueError("incorrect packed index length")
    if bit_count % 8 and int(packed[-1]) >> (bit_count % 8):
        raise ValueError("nonzero index padding")
    if width == 0:
        return np.zeros(count, dtype=np.int64)
    bits = np.unpackbits(packed, bitorder="little")[:bit_count].reshape(count, width)
    labels = np.zeros(count, dtype=np.uint64)
    for bit in range(width):
        labels |= bits[:, bit].astype(np.uint64) << bit
    if np.any(labels >= num_centers):
        raise ValueError("codebook index out of range")
    return labels.astype(np.int64)


class ModelCodec:
    """Fixed-architecture codec shared by clients and server.

    ``parameter_names`` must come from ``model.named_parameters()`` with
    ``requires_grad`` true; never infer them from floating state_dict entries.
    All outputs own their arrays. No operation mutates the supplied state.
    """

    def __init__(self, reference_state: Mapping, parameter_names: Iterable[str]):
        if not isinstance(reference_state, Mapping) or not reference_state:
            raise ValueError("reference_state must be a nonempty mapping")
        self.names = tuple(reference_state)
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("state names must be nonempty strings")
        requested = tuple(parameter_names)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("parameter_names must be nonempty and unique")
        if not set(requested).issubset(self.names):
            raise ValueError("unknown parameter name")
        selected = set(requested)
        self.parameter_names = tuple(name for name in self.names if name in selected)
        self.buffer_names = tuple(name for name in self.names if name not in selected)
        self._layout = {
            name: (np.asarray(reference_state[name]).shape, np.asarray(reference_state[name]).dtype)
            for name in self.names
        }
        for name, (_, dtype) in self._layout.items():
            if dtype.kind not in "fiub" or (name in selected and dtype.kind != "f"):
                raise ValueError(f"unsupported state dtype: {name}")
            if dtype.kind == "f" and dtype.itemsize > 8:
                raise ValueError("floating state must be float16, float32 or float64")
        self.parameter_count = sum(np.asarray(reference_state[n]).size for n in self.parameter_names)
        if not self.parameter_count:
            raise ValueError("model has no trainable parameter values")
        self.center_dtype = np.dtype(np.result_type(np.float32, *[
            self._layout[name][1] for name in self.parameter_names
        ]))
        layout = [(n, self._layout[n][0], self._layout[n][1].str, n in selected) for n in self.names]
        signature = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
        self._headers = {
            kind: np.frombuffer(f"fedcode-v1:{signature}:{kind}".encode(), dtype=np.uint8).copy()
            for kind in ("raw", "calibration", "codebook")
        }
        self._validate_state(reference_state)

    def _validate_array(self, name, value):
        value = np.asarray(value)
        shape, dtype = self._layout[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"state shape/dtype mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite model state: {name}")
        if (name == "running_var" or name.endswith(".running_var")) and np.any(value < 0):
            raise ValueError(f"negative BatchNorm running variance: {name}")
        return value

    def _validate_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.names):
            raise ValueError("model state names do not match codec schema")
        for name in self.names:
            self._validate_array(name, state[name])

    def _validate_centers(self, centers):
        centers = np.asarray(centers)
        if (centers.ndim != 1 or not 1 <= centers.size <= self.parameter_count
                or centers.dtype != self.center_dtype or not np.isfinite(centers).all()):
            raise ValueError("invalid codebook shape/dtype/values")
        if np.any(centers[1:] <= centers[:-1]):
            raise ValueError("codebook centers must be sorted and unique")
        return centers

    def cluster(self, state, k, seed=0, *, max_iter=300, n_init=10):
        """Deterministic global scalar K-means (k-means++, 10 starts, 300 steps).

        Fewer than k distinct values produce a smaller, lossless codebook.
        Centers are canonicalized after casting to the wire precision.
        """
        self._validate_state(state)
        k = _integer(k, "k", 1)
        seed = _integer(seed, "seed")
        max_iter = _integer(max_iter, "max_iter", 1)
        n_init = _integer(n_init, "n_init", 1)
        if seed >= 2**32 or k > self.parameter_count:
            raise ValueError("seed or number of clusters out of range")
        values = np.concatenate([np.asarray(state[n]).reshape(-1) for n in self.parameter_names])
        unique = np.unique(values)
        if unique.size <= k:
            centers = unique.astype(self.center_dtype)
        else:
            with threadpool_limits(limits=1):
                fit = KMeans(n_clusters=k, init="k-means++", max_iter=max_iter,
                             n_init=n_init, random_state=seed, algorithm="lloyd").fit(
                                 values.astype(np.float64).reshape(-1, 1))
            centers = np.unique(fit.cluster_centers_.reshape(-1).astype(self.center_dtype))
        self._validate_centers(centers)
        return centers.copy(), _nearest(values, centers)

    def raw(self, state):
        self._validate_state(state)
        return [self._headers["raw"].copy(), *[np.asarray(state[n]).copy() for n in self.names]]

    def calibration(self, state, centers, labels):
        self._validate_state(state)
        centers = self._validate_centers(centers)
        labels = np.asarray(labels)
        if labels.shape != (self.parameter_count,):
            raise ValueError("incorrect parameter index count")
        packed = pack_indices(labels, centers.size)
        return [self._headers["calibration"].copy(), centers.copy(), packed,
                *[np.asarray(state[n]).copy() for n in self.buffer_names]]

    def codebook(self, state, centers):
        self._validate_state(state)
        centers = self._validate_centers(centers)
        return [self._headers["codebook"].copy(), centers.copy(),
                *[np.asarray(state[n]).copy() for n in self.buffer_names]]

    def _inspect(self, payload):
        if not isinstance(payload, Sequence) or not len(payload):
            raise ValueError("payload must be a nonempty array sequence")
        header = np.asarray(payload[0])
        if header.dtype != np.uint8 or header.ndim != 1:
            raise ValueError("invalid payload header")
        kind = next((k for k, h in self._headers.items() if np.array_equal(header, h)), None)
        if kind is None:
            raise ValueError("unknown payload format or model schema")
        if kind == "raw":
            if len(payload) != len(self.names) + 1:
                raise ValueError("incorrect raw tensor count")
            for name, value in zip(self.names, payload[1:]):
                self._validate_array(name, value)
            return kind, None, None
        offset = 3 if kind == "calibration" else 2
        if len(payload) != len(self.buffer_names) + offset:
            raise ValueError("incorrect compressed tensor count")
        centers = self._validate_centers(payload[1])
        labels = unpack_indices(payload[2], self.parameter_count, len(centers)) if offset == 3 else None
        for name, value in zip(self.buffer_names, payload[offset:]):
            self._validate_array(name, value)
        return kind, centers, labels

    def kind(self, payload):
        """Return payload mode after validating the entire packet."""
        return self._inspect(payload)[0]

    def centers(self, payload):
        centers = self._inspect(payload)[1]
        if centers is None:
            raise ValueError("raw payload has no codebook")
        return centers.copy()

    def apply(self, payload, reference=None):
        """Decode raw/calibration state or remap a stored state using a codebook."""
        kind, centers, labels = self._inspect(payload)
        if kind == "raw":
            return OrderedDict((n, np.asarray(v).copy()) for n, v in zip(self.names, payload[1:]))
        if kind == "codebook":
            self._validate_state(reference)
        buffers = dict(zip(self.buffer_names, payload[3 if kind == "calibration" else 2:]))
        state, offset = OrderedDict(), 0
        for name in self.names:
            shape, dtype = self._layout[name]
            if name in buffers:
                state[name] = np.asarray(buffers[name]).copy()
            else:
                count = int(np.prod(shape, dtype=np.int64))
                indices = labels[offset:offset + count] if labels is not None else _nearest(
                    np.asarray(reference[name]).reshape(-1), centers)
                state[name] = centers[indices].reshape(shape).astype(dtype, copy=True)
                offset += count
        self._validate_state(state)
        return state

    @staticmethod
    def _counts(counts, length):
        if len(counts) != length or not length:
            raise ValueError("updates and sample counts must be nonempty and aligned")
        counts = [_integer(n, "sample count", 1) for n in counts]
        total = sum(counts)
        return [n / total for n in counts]

    def _average_names(self, states, counts, names):
        weights = self._counts(counts, len(states))
        result = OrderedDict()
        for name in names:
            shape, dtype = self._layout[name]
            if dtype.kind == "f":
                value = np.zeros(shape, dtype=np.float64)
                for state, weight in zip(states, weights):
                    value += np.asarray(state[name], dtype=np.float64) * weight
                result[name] = value.astype(dtype)
            else:
                # BatchNorm counters are metadata, not real-valued model weights.
                result[name] = np.asarray(np.maximum.reduce([np.asarray(s[name]) for s in states])).copy()
        return result

    def average(self, states, counts):
        """Sample-weighted FedAvg; integer/bool metadata use elementwise max."""
        for state in states:
            self._validate_state(state)
        result = self._average_names(states, counts, self.names)
        self._validate_state(result)
        return result

    def aggregate_codebooks(self, reference, payloads, counts):
        """Nearest union for parameters, sample-weighted auxiliary statistics.

        Parameter centers are concatenated without sample weighting, as in the
        paper. The buffer policy is explicit engineering needed by state_dict.
        """
        self._validate_state(reference)
        self._counts(counts, len(payloads))
        books, buffers = [], []
        for payload in payloads:
            kind, centers, _ = self._inspect(payload)
            if kind != "codebook":
                raise ValueError("codebook aggregation requires codebook-only packets")
            books.append(centers)
            buffers.append(dict(zip(self.buffer_names, payload[2:])))
        union = np.unique(np.concatenate(books))
        result = self._average_names(buffers, counts, self.buffer_names)
        for name in self.parameter_names:
            shape, dtype = self._layout[name]
            result[name] = union[_nearest(np.asarray(reference[name]).reshape(-1), union)].reshape(shape).astype(dtype)
        result = OrderedDict((n, result[n]) for n in self.names)
        self._validate_state(result)
        return result

    def payload_bytes(self, payload):
        """All NumPy payload bytes, including header/buffers; excludes RPC framing."""
        self._inspect(payload)
        return sum(np.asarray(value).nbytes for value in payload)
