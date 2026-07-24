"""Chapter 11: Residual connections and layer normalisation.

    out = LayerNorm(x + Sublayer(x))

Two quiet inventions that make deep stacks trainable at all. The residual
is the "+ x": a wire that carries the original through untouched, so a
sublayer only has to learn a correction. Layer norm then puts every row
back on one common scale.

    python part2_transformer/ch11_layernorm.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specimen import LN1_BETA, LN1_GAMMA  # noqa: E402


def layernorm(x, gamma, beta, eps=1e-5):
    """Normalise ONE row, on its own.

    No axis argument, because there is nothing to choose: layer norm looks
    at a single row. That is exactly what makes it independent of the batch
    (and therefore safe at batch size 1, unlike batch norm).
    """
    mu = x.mean()
    sd = np.sqrt(x.var() + eps)
    return gamma * (x - mu) / sd + beta


if __name__ == "__main__":
    # sat's row as it leaves self-attention, and the change attention proposed
    x = np.array([1.4081, -0.1314, 1.0616, 1.6150])
    sublayer_out = np.array([-0.0015, 0.0159, 0.1087, 0.0357])

    print("input x            :", np.round(x, 4))
    print("Sublayer(x)        :", np.round(sublayer_out, 4))
    print("  ^ notice how small the correction is next to the input.")
    print("    The sublayer nudges the word; it does not rebuild it.")

    residual = x + sublayer_out
    print("\nx + Sublayer(x)    :", np.round(residual, 4))

    print("\nmean               : %.4f   (the centre we subtract)" % residual.mean())
    print("std                : %.4f   (the spread we divide by)" % np.sqrt(residual.var()))

    out = layernorm(residual, LN1_GAMMA, LN1_BETA)
    print("\nLayerNorm(...)     :", np.round(out, 4))
    print("book prints        : [ 0.3967 -1.9573  0.3149  0.9487]")

    plain = (residual - residual.mean()) / np.sqrt(residual.var() + 1e-5)
    print("\nBefore gamma and beta, the normalised row is")
    print("  ", np.round(plain, 4))
    print("  mean %.6f, std %.4f: the clean slate." % (plain.mean(), plain.std()))
    print("  gamma and beta are the ONLY learned numbers here;")
    print("  the normalising itself has no parameters at all.")
