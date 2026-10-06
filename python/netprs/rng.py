"""Portable pseudo-random streams shared by the MATLAB and Python versions.

The generator is a bank of 64 xorshift64 lanes (Marsaglia, 2003; shift triple
13, 7, 17). It uses only 64-bit shifts and exclusive-or, which behave
identically in MATLAB (``uint64`` with ``bitshift``/``bitxor``), GNU Octave and
NumPy, so a given ``(seed, stream)`` pair yields bit-identical numbers in all
three environments. NetPRS uses it for the synthetic data, the stratified
cross-validation folds and the initialization of ``beta``; MATLAB and Python
therefore train the same models from the same starting point.

Specification (mirrored by ``RngStream.m``, ``RngUniform.m``,
``RngPermutation.m``)
---------------------------------------------------------------------------
* Lane constants: ``c[0] = 0x9E3779B97F4A7C15`` and ``c[l+1] = xorshift(c[l])``.
* ``init(seed, stream)``: ``key = seed | (stream << 32)``,
  ``state[l] = c[l] ^ key`` (a zero state is replaced by ``c[l]``), followed by
  64 warm-up steps of every lane.
* ``uniform(n)``: each step advances all lanes once and emits, lane by lane,
  ``(state >> 11) * 2**-53`` (53-bit uniforms in [0, 1)); whole steps are
  always consumed and the first ``n`` numbers are returned.
* ``permutation(n)``: stable ``argsort`` of ``uniform(n)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RandomStream", "NUM_LANES"]

NUM_LANES = 64
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_S13, _S7, _S17, _S11 = (np.uint64(k) for k in (13, 7, 17, 11))
_SCALE = 2.0 ** -53


def _xorshift(x: np.ndarray) -> np.ndarray:
    """One xorshift64 step (13, 7, 17) applied element-wise to uint64 lanes."""
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
    """Portable random stream identified by ``(seed, stream)``.

    Parameters
    ----------
    seed : int
        Integer in ``[0, 2**32)``.
    stream : int, default 0
        Integer in ``[0, 2**16)`` separating independent uses of one seed.
    """

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
        """Return ``n`` uniforms in [0, 1) (float64, shape ``(n,)``)."""
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
        """Return a random permutation of ``0, ..., n-1`` (int64)."""
        return np.argsort(self.uniform(n), kind="stable").astype(np.int64)
