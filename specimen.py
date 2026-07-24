"""The trained model's real numbers: the single source of truth.

Every other file in this repo imports from here, and nothing re-types a
matrix. This mirrors the discipline the book itself uses: the printed
numbers are exported from a real trained transformer, never hand-typed,
which is what makes "check it yourself" an honest promise.

The model is deliberately tiny so it fits on a page:
    3 tokens ("the cat sat"), d_model = 4, 2 heads, so d_k = 2.
It translates English -> Hindi: "the cat sat" -> "billi baithi".

One caveat worth reading once. These matrices are rounded to 4 decimal
places, because that is how they are printed in the book. The model
carries full precision internally. So a value you recompute here can
differ from the book in the last digit (you will see 1.6134 where the
book prints 1.6135). That gap IS rounding error, and the book calls it
out on the same page rather than hiding it.
"""

import numpy as np

# ---- the running specimen -------------------------------------------
TOKENS = ["the", "cat", "sat"]
TRANSLATION = "billi baithi"

D_MODEL = 4
N_HEADS = 2
D_K = 2

# ---- input: one row per word, embedding + position ------------------
X = np.array([
    [-0.1306,  1.1230, -0.3542,  0.6248],   # the
    [-0.1545,  0.6844, -0.6097,  1.3545],   # cat
    [ 1.4081, -0.1314,  1.0616,  1.6150],   # sat
])

# ---- learned projection weights (d_model x d_model) -----------------
W_Q = np.array([
    [-0.7686,  0.0609, -0.4290,  0.1220],
    [ 0.5499, -0.4918, -0.1216,  0.0922],
    [-0.3904,  0.7040,  0.0173,  0.5663],
    [ 0.0800,  0.2012,  0.0594,  0.3569],
])
W_K = np.array([
    [ 0.0268,  0.0272,  0.3304, -0.2843],
    [ 0.0949, -0.4809, -0.2237,  0.0184],
    [ 0.1325, -0.1269,  0.0145, -0.3610],
    [-0.6109,  0.4265,  0.0715, -0.5135],
])
W_V = np.array([
    [ 0.1945, -0.3746, -0.1853,  0.1116],
    [ 0.2690,  0.2405,  0.3027,  0.2123],
    [-0.5743,  0.2521,  0.3633,  0.3520],
    [-0.0824,  0.1932, -0.0449, -0.1051],
])
W_O = np.array([
    [-0.7098,  0.5035, -0.5955, -0.3006],
    [-0.2532,  0.1228,  0.3511, -0.0575],
    [ 0.1171, -0.0584,  0.0635,  0.3412],
    [-0.1460, -0.5015, -0.4015, -0.0766],
])

# ---- layer norm parameters for the first sublayer (ch 11) -----------
LN1_GAMMA = np.array([0.8285, 1.2019, 0.8412, 1.1171])
LN1_BETA = np.array([-0.0634, 0.0588, 0.1394, -0.0717])


def head_slice(M, head):
    """Columns belonging to one head. Head 0 is the one the book follows."""
    return M[:, head * D_K:(head + 1) * D_K]


def qkv(head=0):
    """Q, K, V for one head, straight from X and the learned weights."""
    Q = head_slice(X @ W_Q, head)
    K = head_slice(X @ W_K, head)
    V = head_slice(X @ W_V, head)
    return Q, K, V
