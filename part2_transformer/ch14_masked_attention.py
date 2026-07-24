"""Chapter 14: the decoder, and writing with a blindfold on the future.

When the model writes word t, words t+1 onward do not exist yet. So we
forbid position t from attending to them: set those scores to -inf BEFORE
the softmax. Since exp(-inf) = 0, they get exactly zero weight, and the
surviving weights still sum to 1 on their own.

    python part2_transformer/ch14_masked_attention.py
"""

import numpy as np


def causal_mask(S):
    """Hide the future. k=1 means strictly ABOVE the diagonal.

    With k=0 the diagonal would be masked too, and every position would be
    forbidden from seeing itself. That is not caution, it is amnesia.
    """
    future = np.triu(np.ones(S.shape, dtype=bool), k=1)
    return np.where(future, -np.inf, S)


def softmax_rows(S):
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    words = ["<bos>", "billi", "baithi"]
    S = np.array([[2.0, 1.0, 0.5],
                  [0.3, 1.7, 1.1],
                  [1.2, 0.4, 2.3]])

    print("raw scores")
    print(np.round(S, 3))

    print("\nafter the causal mask (-inf above the diagonal)")
    print(causal_mask(S))

    A = softmax_rows(causal_mask(S))
    print("\nattention weights")
    for w, row in zip(words, A):
        allowed = "  ".join("%-6s %.3f" % (t, v) for t, v in zip(words, row))
        print("  %-7s -> %s" % (w, allowed))

    print("\nTwo things to notice.")
    print("  upper triangle is exactly zero :", bool((np.triu(A, k=1) == 0).all()))
    print("  yet every row still sums to 1  :", np.round(A.sum(axis=1), 6))
    print("\nBlocking the future does not leave a position with less")
    print("attention to spend. It forces all of it onto the past.")
