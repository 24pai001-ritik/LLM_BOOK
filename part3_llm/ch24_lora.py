"""Chapter 24: LoRA. Fine-tuning without the pain.

Instead of updating a d x d weight matrix, learn a low-rank correction:

    W' = W + B A        with A: r x d,  B: d x r,  r << d

Train A and B, freeze W. The saving is not subtle, and this file counts it.

    python part3_llm/ch24_lora.py
"""

import numpy as np


def lora_update(W, A, B, alpha=1.0):
    """The frozen matrix plus a scaled low-rank correction."""
    r = A.shape[0]
    return W + (alpha / r) * (B @ A)


def parameter_counts(d, r):
    full = d * d
    lora = 2 * d * r          # A is r x d, B is d x r
    return full, lora


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, r = 8, 2

    W = rng.normal(0, 0.1, (d, d))
    A = rng.normal(0, 0.01, (r, d))
    B = np.zeros((d, r))      # B starts at ZERO on purpose

    print("At initialisation B = 0, so B @ A = 0 and W' == W exactly.")
    print("  max |W' - W| at step 0 : %.1e" % np.abs(lora_update(W, A, B) - W).max())
    print("  The adapted model therefore STARTS as the original model,")
    print("  which is why LoRA training never begins with a quality drop.\n")

    B = rng.normal(0, 0.01, (d, r))     # after a little training
    delta = lora_update(W, A, B) - W
    print("after training, the correction has rank %d (not %d):"
          % (np.linalg.matrix_rank(delta), d))
    print("  singular values:", np.round(np.linalg.svd(delta, compute_uv=False), 5))
    print("  only %d are non-zero, which is the whole trick.\n" % r)

    print("parameters you must train, per d x d matrix:")
    print("  %-8s %-14s %-14s %s" % ("d", "full", "LoRA r=8", "saving"))
    for dim in (768, 1024, 4096, 8192):
        full, lora = parameter_counts(dim, 8)
        print("  %-8d %-14s %-14s %5.1fx"
              % (dim, "{:,}".format(full), "{:,}".format(lora), full / lora))

    print("\nAnd the memory that actually stops you fine-tuning a 70B model:")
    print("  full fine-tune needs weights + gradients + 2 Adam moments")
    print("  70e9 params x 4 bytes x 4 copies = %.0f GB" % (70e9 * 4 * 4 / 1e9))
    print("  which is why it needs a data centre, and LoRA does not.")
