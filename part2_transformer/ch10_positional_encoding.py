"""Chapter 10: where am I? Positional encoding.

Attention is permutation equivariant: shuffle the words and it returns the
same vectors, shuffled. So position must be ADDED to the input, or "the cat
sat" and "sat cat the" are literally the same input.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    python part2_transformer/ch10_positional_encoding.py
"""

import numpy as np


def positional_encoding(n_positions, d_model):
    pos = np.arange(n_positions)[:, None]            # column of positions
    i = np.arange(d_model)[None, :]                  # row of dimensions
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])             # even dims: sine
    pe[:, 1::2] = np.cos(angle[:, 1::2])             # odd dims: cosine
    return pe


if __name__ == "__main__":
    pe = positional_encoding(6, 4)
    print("positional encodings, 6 positions x d_model 4")
    for p, row in enumerate(pe):
        print("  pos %d : %s" % (p, np.round(row, 4)))

    print("\nWhy sinusoids and not just the integer 0,1,2,...?")
    print("Because RELATIVE distance stays readable. The dot product")
    print("between two position vectors depends mostly on the GAP:")
    for gap in (1, 2, 3):
        sims = [pe[p] @ pe[p + gap] for p in range(6 - gap)]
        print("  gap %d : dot products %s" % (gap, np.round(sims, 3)))
    print("\nSame gap gives a similar value wherever you are in the")
    print("sentence, so 'two words apart' means one thing at position 0")
    print("and the same thing at position 40.")
