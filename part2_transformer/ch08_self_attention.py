"""Chapter 8: Attention, from the ground up.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

That one line is the whole chapter, and the whole transformer. Run this
file and watch the word "sat" decide, on its own, that it belongs with
"cat" and not with the filler "the".

    python part2_transformer/ch08_self_attention.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specimen import TOKENS, qkv  # noqa: E402


def softmax(z):
    """Turn a row of scores into a row of fractions that sum to 1.

    Subtracting the max changes no answer (it cancels in the ratio) but
    keeps exp() away from overflow. Every real implementation does this.
    """
    e = np.exp(z - z.max())
    return e / e.sum()


def attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)          # how well each query matches each key
    weights = np.array([softmax(row) for row in scores])
    return weights @ V, weights              # blend the values


if __name__ == "__main__":
    Q, K, V = qkv(head=0)
    out, weights = attention(Q, K, V)

    print("raw scores  Q K^T")
    print(np.round(Q @ K.T, 4))
    print("\nscaled by sqrt(d_k) = sqrt(2)")
    print(np.round(Q @ K.T / np.sqrt(2), 4))

    print("\nattention weights (each row sums to 1)")
    for tok, row in zip(TOKENS, weights):
        share = "  ".join("%s %.4f" % (t, w) for t, w in zip(TOKENS, row))
        print("  %-4s attends to:  %s" % (tok, share))

    print("\nThe payoff. Query 'sat' gives:")
    the, cat, sat = weights[2]
    print("  'cat' a %.4f share, but the filler 'the' only %.4f." % (cat, the))
    print("  That is %.1fx more attention on the subject than on the filler," % (cat / the))
    print("  and nobody ever told the model which word was the subject.")

    print("\nsat's new vector:", np.round(out[2], 4))
