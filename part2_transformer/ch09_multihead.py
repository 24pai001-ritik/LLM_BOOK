"""Chapter 9: many eyes. Multi-head attention.

One head learns one kind of relationship. Split the 4 columns into 2 heads
of 2, run attention in each independently, concatenate, and mix with W_O.
Same equation as chapter 8, run twice on narrower slices.

    python part2_transformer/ch09_multihead.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specimen import N_HEADS, TOKENS, W_O, qkv  # noqa: E402
from ch08_self_attention import attention  # noqa: E402


def multihead(n_heads=N_HEADS):
    outs, all_weights = [], []
    for h in range(n_heads):
        Q, K, V = qkv(head=h)
        out, w = attention(Q, K, V)
        outs.append(out)
        all_weights.append(w)
    concat = np.hstack(outs)      # glue the heads back side by side
    return concat @ W_O, concat, all_weights


if __name__ == "__main__":
    out, concat, weights = multihead()

    for h, w in enumerate(weights):
        print("head %d, what each word attends to" % h)
        for tok, row in zip(TOKENS, w):
            print("  %-4s %s" % (tok, "  ".join("%s %.4f" % (t, v)
                                                for t, v in zip(TOKENS, row))))
        print()

    print("The heads disagree, and that is the entire point.")
    print("  head 0, 'sat' -> 'cat' : %.4f" % weights[0][2][1])
    print("  head 1, 'sat' -> 'cat' : %.4f" % weights[1][2][1])
    print("\nconcatenated (3 x 4), then mixed by W_O:")
    print(np.round(out, 4))
