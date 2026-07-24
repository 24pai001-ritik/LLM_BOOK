"""Check that this code reproduces the numbers printed in the book.

The book's central promise is that every worked number is real and you can
check it. This file is that promise, executable. It runs the core paths and
asserts the results against the values printed on the page.

    python verify.py

If it prints ALL CHECKS PASSED, the book and this repo agree.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "part1_language"))
sys.path.insert(0, os.path.join(HERE, "part2_transformer"))
sys.path.insert(0, os.path.join(HERE, "part3_llm"))

from specimen import LN1_BETA, LN1_GAMMA, qkv                    # noqa: E402
from ch08_self_attention import attention, softmax               # noqa: E402
from ch11_layernorm import layernorm                             # noqa: E402
from ch14_masked_attention import causal_mask, softmax_rows      # noqa: E402

checks = []


def check(name, got, want, page, tol=1e-4):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    ok = got.shape == want.shape and np.abs(got - want).max() <= tol
    checks.append(ok)
    status = "ok  " if ok else "FAIL"
    print("  [%s] %-46s (book p.%s)" % (status, name, page))
    if not ok:
        print("         got  %s" % np.round(got, 4))
        print("         want %s" % want)


print("Chapter 8: self-attention on 'the cat sat'")
Q, K, V = qkv(head=0)
check("Q, head 0", Q,
      [[0.9062, -0.6839], [0.8415, -0.5027], [-1.4398, 1.2227]], 54)
check("K, head 0", K,
      [[-0.3255, -0.2322], [-0.8474, 0.3217], [-0.8207, 0.6556]], 54)
check("V, head 0", V,
      [[0.4286, 0.3504], [0.3926, 0.3305], [-0.5042, 0.0206]], 54)
check("raw scores Q K^T, row 'sat'", (Q @ K.T)[2], [0.1848, 1.6135, 1.9831], 55)
check("scaled by sqrt(2), row 'sat'", (Q @ K.T / np.sqrt(2))[2],
      [0.1307, 1.1409, 1.4023], 57)

out, weights = attention(Q, K, V)
check("attention weights, row 'sat'", weights[2], [0.1367, 0.3755, 0.4877], 58)
check("output vector for 'sat'", out[2], [-0.0399, 0.1821], 59)

print("\nChapter 11: residual + layer norm")
residual = np.array([1.4081, -0.1314, 1.0616, 1.6150]) + \
           np.array([-0.0015, 0.0159, 0.1087, 0.0357])
check("x + Sublayer(x), row 'sat'", residual,
      [1.4066, -0.1155, 1.1703, 1.6507], 76)
check("mean of that row", residual.mean(), 1.0280, 78)
check("std of that row", np.sqrt(residual.var()), 0.6817, 78)
check("LayerNorm output, row 'sat'",
      layernorm(residual, LN1_GAMMA, LN1_BETA),
      [0.3967, -1.9573, 0.3149, 0.9487], 78, tol=2e-4)

print("\nChapter 14: the causal mask")
S = np.array([[2.0, 1.0, 0.5], [0.3, 1.7, 1.1], [1.2, 0.4, 2.3]])
A = softmax_rows(causal_mask(S))
check("future is exactly zero", np.triu(A, k=1), np.zeros((3, 3)), 95)
check("every row still sums to 1", A.sum(axis=1), [1.0, 1.0, 1.0], 95)

print("\nProperties that must hold everywhere")
check("softmax sums to 1", softmax(np.array([3.0, 1.0, 0.2])).sum(), 1.0, "-")
check("softmax is shift invariant",
      softmax(np.array([3.0, 1.0, 0.2])),
      softmax(np.array([3.0, 1.0, 0.2]) + 100.0), "-")
check("attention rows sum to 1", weights.sum(axis=1), [1.0, 1.0, 1.0], "-")

print("\n" + "-" * 66)
if all(checks):
    print("ALL CHECKS PASSED  (%d/%d)" % (len(checks), len(checks)))
    print("The code and the book agree, to the last printed digit.")
else:
    print("FAILURES: %d of %d" % (checks.count(False), len(checks)))
    sys.exit(1)
