"""Chapter 3: counting the future. A bigram language model.

Before neural anything, a language model was a counting exercise:

    P(next | current) = count(current, next) / count(current)

This is the whole idea, and it also shows you exactly why it breaks: a
pair never seen in training gets probability zero, forever.

    python part1_language/ch03_ngram.py
"""

from collections import Counter, defaultdict
import math

CORPUS = """the cat sat on the mat
the dog sat on the rug
the cat ran to the mat
a dog ran to the park
the cat sat on the rug""".strip().split("\n")


def train(corpus):
    """Count every (word, next word) pair. That is the entire training run."""
    pairs = Counter()
    firsts = Counter()
    for line in corpus:
        words = line.split()
        for a, b in zip(words, words[1:]):
            pairs[(a, b)] += 1
            firsts[a] += 1
    return pairs, firsts


def probability(pairs, firsts, a, b):
    if firsts[a] == 0:
        return 0.0
    return pairs[(a, b)] / firsts[a]


def perplexity(pairs, firsts, sentence):
    """exp(average surprise). Lower is better. Infinite means 'impossible'."""
    words = sentence.split()
    total = 0.0
    for a, b in zip(words, words[1:]):
        p = probability(pairs, firsts, a, b)
        if p == 0:
            return float("inf")
        total += -math.log(p)
    return math.exp(total / (len(words) - 1))


if __name__ == "__main__":
    pairs, firsts = train(CORPUS)

    print("what usually follows 'the'?")
    after = defaultdict(float)
    for (a, b), n in pairs.items():
        if a == "the":
            after[b] = n / firsts["the"]
    for w, p in sorted(after.items(), key=lambda kv: -kv[1]):
        print("  the %-6s %.2f  %s" % (w, p, "#" * int(p * 40)))

    print("\nP(sat | cat) = %.2f" % probability(pairs, firsts, "cat", "sat"))
    print("P(ran | cat) = %.2f" % probability(pairs, firsts, "cat", "ran"))

    print("\nperplexity, a sentence it has seen the shape of:")
    print("  'the cat sat on the mat'  -> %.2f"
          % perplexity(pairs, firsts, "the cat sat on the mat"))
    print("\nand now the fatal flaw:")
    print("  'the cat flew to the moon' -> %s"
          % perplexity(pairs, firsts, "the cat flew to the moon"))
    print("\nInfinite. Not 'unlikely', but impossible: one unseen pair and")
    print("the whole sentence is ruled out. No amount of counting fixes")
    print("this, which is why chapter 4 goes looking for vectors instead.")
