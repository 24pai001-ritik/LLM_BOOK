"""Chapter 18: byte-pair encoding, the tokenizer behind almost every LLM.

Start from characters. Repeatedly find the most frequent adjacent pair and
merge it into one new symbol. Common words collapse into single tokens;
rare words stay in pieces. Nothing is ever out-of-vocabulary.

    python part3_llm/ch18_bpe.py
"""

from collections import Counter

CORPUS = ("low low low low low lower lower newest newest newest "
          "newest newest newest widest widest widest")


def get_pairs(words):
    pairs = Counter()
    for word, freq in words.items():
        symbols = word.split()
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge(words, pair):
    """Glue one pair together everywhere it occurs."""
    bigram = " ".join(pair)
    joined = "".join(pair)
    return {w.replace(bigram, joined): f for w, f in words.items()}


def train_bpe(corpus, n_merges):
    # "</w>" marks a word end, so "low" and "lower" can share a prefix
    words = Counter(corpus.split())
    words = {" ".join(w) + " </w>": f for w, f in words.items()}
    merges = []
    for step in range(n_merges):
        pairs = get_pairs(words)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        words = merge(words, best)
        merges.append((best, pairs[best]))
        print("  merge %2d : %-14s (seen %d times)"
              % (step + 1, "'%s' + '%s'" % best, pairs[best]))
    return words, merges


if __name__ == "__main__":
    print("learning merges from a tiny corpus\n")
    words, merges = train_bpe(CORPUS, 10)

    print("\nfinal segmentation:")
    for w, f in sorted(words.items(), key=lambda kv: -kv[1]):
        print("  %-22s x%d" % (w, f))

    print("\nNotice what happened. 'low' became a single token because it")
    print("is common. 'widest' stayed in pieces because it is not. That is")
    print("the whole trade: frequent things get short codes, and anything")
    print("unseen can still be spelled out character by character, so the")
    print("model is never handed a word it cannot represent at all.")
