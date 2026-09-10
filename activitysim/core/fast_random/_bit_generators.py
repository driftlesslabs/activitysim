"""PCG64 and SFC64 transitions over owned uint64 arrays, without NumPy's C ABI.

Seeding uses NumPy's public state dictionaries. Arithmetic in these compiled
kernels intentionally wraps modulo 2**64. The transitions below mirror:
https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/pcg64/pcg64.h
https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/sfc64/sfc64.h
"""

# PCG transition and multiplication adapted under the MIT license:
# Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
# Copyright 2015 Robert Kern <robert.kern@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Copyright (c) 2005-2025, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import annotations

import numba as nb
import numpy as np
from llvmlite import ir
from numba.extending import intrinsic


@intrinsic
def _multiply_high(typingctx, a, b):
    """Return the high 64 bits of an unsigned 64-by-64-bit product."""
    if a != nb.types.uint64 or b != nb.types.uint64:
        return None

    def codegen(context, builder, signature, args):
        # Widen before multiplying so LLVM can select a native multiply-high
        # instruction (e.g. ARM64 UMULH). Splitting into 32-bit partial products
        # obscures this operation and slows long PCG64 uniform sequences.
        # LLVM handles target lowering; this needs no architecture-specific ASM
        # or access to NumPy's private state layout.
        wide = ir.IntType(128)
        product = builder.mul(builder.zext(args[0], wide), builder.zext(args[1], wide))
        high = builder.lshr(product, ir.Constant(wide, 64))
        return builder.trunc(high, ir.IntType(64))

    return nb.types.uint64(a, b), codegen


@nb.njit(inline="always")
def pcg64_next_uint64(state):
    """Advance PCG64 XSL RR; words are state low/high, increment low/high."""
    multiplier_low = np.uint64(4865540595714422341)
    multiplier_high = np.uint64(2549297995355413924)
    low, high = state[0], state[1]
    product_low = low * multiplier_low
    new_low = product_low + state[2]
    new_high = (
        _multiply_high(low, multiplier_low)
        + high * multiplier_low
        + low * multiplier_high
        + state[3]
        + np.uint64(new_low < product_low)
    )
    state[0], state[1] = new_low, new_high
    value = new_low ^ new_high
    rotation = new_high >> np.uint64(58)
    # Mask both shifts to avoid an undefined shift by 64 when rotation is zero.
    return (value >> rotation) | (value << ((np.uint64(64) - rotation) & np.uint64(63)))


@nb.njit(inline="always")
def sfc64_next_uint64(state):
    """Advance SFC64; words are a, b, c, counter as in NumPy's public state."""
    value = state[0] + state[1] + state[3]
    state[3] += np.uint64(1)
    a, b = state[1], state[2]
    state[0] = a ^ (a >> np.uint64(11))
    state[1] = b + (b << np.uint64(3))
    state[2] = ((b << np.uint64(24)) | (b >> np.uint64(40))) + value
    return value


@nb.njit(inline="always")
def pcg64_next_double(state):
    """Convert the high 53 bits to a uniform double, matching NumPy."""
    return (pcg64_next_uint64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(inline="always")
def sfc64_next_double(state):
    """Convert the high 53 bits to a uniform double, matching NumPy."""
    return (sfc64_next_uint64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)
