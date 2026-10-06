from __future__ import annotations

import numpy as np

__all__ = ["RandomStream", "NUM_LANES"]

NUM_LANES = 64
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_S13, _S7, _S17, _S11 = (np.uint64(k) for k in (13, 7, 17, 11))
_SCALE = 2.0 ** -53


def _xorshift(x: np.ndarray) -> np.ndarray:

    x = x ^ ((x << _S13) & _MASK)
    x = x ^ (x >> _S7)
    x = x ^ ((x << _S17) & _MASK)
    return x


def _lane_constants() -> np.ndarray:
    c = np.empty(NUM_LANES, dtype=np.uint64)
    c[0] = np.uint64(0x9E3779B97F4A7C15)
    for lane in range(1, NUM_LANES):
        c[lane] = _xorshift(c[lane - 1:lane])[0]
    return c


_LANES = _lane_constants()


class RandomStream:


    def __init__(self, seed: int, stream: int = 0) -> None:
        seed, stream = int(seed), int(stream)
        if not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if not 0 <= stream < 2 ** 16:
            raise ValueError("stream must be an integer in [0, 2**16).")
        key = np.uint64(seed | (stream << 32))
        state = _LANES ^ key
        zero = state == 0
        state[zero] = _LANES[zero]
        for _ in range(64):
            state = _xorshift(state)
        self.state = state

    def uniform(self, n: int) -> np.ndarray:

        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative.")
        steps = -(-n // NUM_LANES)
        out = np.empty((steps, NUM_LANES), dtype=np.float64)
        state = self.state
        for step in range(steps):
            state = _xorshift(state)
            out[step] = (state >> _S11).astype(np.float64) * _SCALE
        self.state = state
        return out.ravel()[:n]

    def permutation(self, n: int) -> np.ndarray:

        return np.argsort(self.uniform(n), kind="stable").astype(np.int64)
