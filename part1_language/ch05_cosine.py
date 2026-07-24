"""Chapter 5: the geometry of meaning.

A word is an arrow. Two words mean similar things when their arrows point
the same way, and "the same way" is measured by the angle between them,
not by how long they are. That is cosine similarity:

    cos(a, b) = (a . b) / (|a| |b|)

    python part1_language/ch05_cosine.py
"""

import numpy as np

# A tiny hand-built space. Each row is a word, each column a "sense"
# dimension. Nothing is learned here; the point is the geometry.
WORDS = ["cat", "dog", "kitten", "car", "truck"]
VECTORS = np.array([
    [0.9, 0.8, 0.1, 0.0],   # cat     : animal, pet, not-vehicle
    [0.9, 0.8, 0.1, 0.0],   # dog     : same profile as cat
    [0.8, 0.9, 0.1, 0.0],   # kitten  : very close to cat
    [0.0, 0.1, 0.9, 0.8],   # car     : vehicle
    [0.1, 0.0, 0.9, 0.9],   # truck   : vehicle
])


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    print("cosine similarity between every pair\n")
    print("        " + "".join("%-9s" % w for w in WORDS))
    for i, w in enumerate(WORDS):
        row = "".join("%-9.2f" % cosine(VECTORS[i], VECTORS[j])
                      for j in range(len(WORDS)))
        print("%-8s%s" % (w, row))

    print("\ncat vs dog    : %.2f" % cosine(VECTORS[0], VECTORS[1]))
    print("cat vs car    : %.2f" % cosine(VECTORS[0], VECTORS[3]))
    print("\nThe distributional hypothesis, in one number: words used in")
    print("the same contexts end up pointing the same way. 'cat' and")
    print("'dog' are near-identical here not because a dictionary said")
    print("so, but because they appear in the same kinds of sentences.")

    print("\nLength does not matter, only direction:")
    print("  cos(cat, cat * 100) = %.2f" % cosine(VECTORS[0], VECTORS[0] * 100))
