"""Codebook + bit-packed index calibration carried by stock Flower ndarrays."""

from collections import OrderedDict
import json
import math
import numpy as np


def floating_weights(state):
    return np.concatenate(
        [
            np.asarray(v).reshape(-1)
            for v in state.values()
            if np.issubdtype(np.asarray(v).dtype, np.floating)
        ]
    ).reshape(-1, 1)


def pack_calibration(state, centers, labels):
    centers = np.asarray(centers).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint64).reshape(-1)
    count = sum(
        np.asarray(v).size
        for v in state.values()
        if np.issubdtype(np.asarray(v).dtype, np.floating)
    )
    if count != len(labels) or not len(centers) or np.any(labels >= len(centers)):
        raise ValueError("Invalid codebook assignment count or index")
    width = max(1, math.ceil(math.log2(len(centers))))
    bits = ((labels[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little")
    layout = [
        {
            "name": k,
            "shape": list(np.asarray(v).shape),
            "dtype": np.asarray(v).dtype.str,
            "clustered": np.issubdtype(np.asarray(v).dtype, np.floating),
        }
        for k, v in state.items()
    ]
    header = {
        "format": "fedcode-calibration-v1",
        "width": width,
        "count": count,
        "layout": layout,
    }
    metadata = np.frombuffer(
        json.dumps(header, separators=(",", ":")).encode(), dtype=np.uint8
    ).copy()
    buffers = [
        np.asarray(v).copy()
        for v in state.values()
        if not np.issubdtype(np.asarray(v).dtype, np.floating)
    ]
    return [metadata, centers.copy(), packed, *buffers]


def is_calibration(payload):
    return len(payload) >= 3 and np.asarray(payload[0]).dtype == np.uint8


def unpack_calibration(payload):
    header = json.loads(np.asarray(payload[0]).tobytes())
    if header.get("format") != "fedcode-calibration-v1":
        raise ValueError("Unsupported calibration format")
    count, width = header["count"], header["width"]
    if not 1 <= width <= 64 or count < 0:
        raise ValueError("Invalid index layout")
    bits = np.unpackbits(np.asarray(payload[2]), bitorder="little")
    if len(bits) < count * width:
        raise ValueError("Truncated calibration indices")
    labels = (
        bits[: count * width].reshape(count, width).astype(np.uint64)
        << np.arange(width, dtype=np.uint64)
    ).sum(axis=1)
    centers = np.asarray(payload[1]).reshape(-1)
    if np.any(labels >= len(centers)):
        raise ValueError("Codebook index out of range")
    weights = centers[labels]
    output = OrderedDict()
    offset = 0
    buffer = 3
    for item in header["layout"]:
        shape = tuple(item["shape"])
        size = math.prod(shape)
        if item["clustered"]:
            value = weights[offset : offset + size]
            offset += size
        else:
            value = np.asarray(payload[buffer])
            buffer += 1
        output[item["name"]] = (
            np.asarray(value, dtype=np.dtype(item["dtype"])).reshape(shape).copy()
        )
    if offset != count or buffer != len(payload):
        raise ValueError("Inconsistent calibration layout")
    return output
