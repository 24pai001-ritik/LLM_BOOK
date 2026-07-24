# The Attention Book — code

Runnable code for **_The Attention Book: Large Language Models from First Principles_**
by Ritik (Ritik Publication).

The book's central promise is that **every worked number in it is real**, computed from a
tiny transformer that was actually trained, and that you can check any of them yourself.
This repository is that promise, executable.

```bash
git clone https://github.com/24pai001-ritik/LLM_BOOK.git
cd LLM_BOOK
pip install -r requirements.txt
python verify.py
```

`verify.py` recomputes the book's headline numbers and asserts them against the values
printed on the page, citing the page number for each. If it says `ALL CHECKS PASSED`, the
code and the book agree to the last printed digit.

## House rules for this code

The code here is written to be **read**, not reused. It is the simplest program that shows
the idea:

- pure Python and NumPy, nothing else;
- no classes, no framework, no config files, no error handling;
- variables are named after the maths on the facing page (`Q`, `K`, `V`, `d_k`, `gamma`),
  so you can put the equation and the code side by side and match them symbol for symbol;
- if a listing needs a comment to explain a trick, the trick is too clever and gets rewritten.

None of this is fast, and none of it should go near production. That is deliberate. A
vectorised, batched, fused implementation teaches you about engineering; this one teaches
you about attention.

## The model these numbers come from

A real encoder-decoder transformer, trained to translate English to Hindi, deliberately
made small enough to fit on a page:

| | |
|---|---|
| sentence | `the cat sat` → `billi baithi` |
| `d_model` | 4 |
| heads | 2, so `d_k` = 2 |
| tokens | 3 |

Every matrix lives in [`specimen.py`](specimen.py), and nothing anywhere else re-types a
number. That is the same single-source-of-truth discipline the book itself uses.

**One honest caveat.** The matrices in `specimen.py` are rounded to 4 decimal places,
because that is how the book prints them. The trained model carries full precision
internally. So a value you recompute can differ in the last digit. That gap *is* rounding
error, and the book says so on the same page rather than quietly hiding it.

## What is where

| File | Chapter | What it shows |
|---|---|---|
| [`specimen.py`](specimen.py) | — | the trained model's real matrices; everything imports from here |
| [`part1_language/ch03_ngram.py`](part1_language/ch03_ngram.py) | 3 | a bigram model by counting, and why zero probability kills it |
| [`part1_language/ch05_cosine.py`](part1_language/ch05_cosine.py) | 5 | meaning as an angle; cosine similarity |
| [`part2_transformer/ch08_self_attention.py`](part2_transformer/ch08_self_attention.py) | 8 | **the flagship**: `softmax(QKᵀ/√d_k)V`, and `sat` finding `cat` |
| [`part2_transformer/ch09_multihead.py`](part2_transformer/ch09_multihead.py) | 9 | two heads, and how they disagree |
| [`part2_transformer/ch10_positional_encoding.py`](part2_transformer/ch10_positional_encoding.py) | 10 | sinusoids, and why relative distance survives |
| [`part2_transformer/ch11_layernorm.py`](part2_transformer/ch11_layernorm.py) | 11 | the residual `+ x` and layer norm |
| [`part2_transformer/ch14_masked_attention.py`](part2_transformer/ch14_masked_attention.py) | 14 | the causal mask: `-inf` before softmax |
| [`part3_llm/ch18_bpe.py`](part3_llm/ch18_bpe.py) | 18 | byte-pair encoding, learned from a tiny corpus |
| [`part3_llm/ch24_lora.py`](part3_llm/ch24_lora.py) | 24 | LoRA's low-rank update, and the parameter saving |
| [`part3_llm/ch26_kv_cache.py`](part3_llm/ch26_kv_cache.py) | 26 | the KV cache bill: 80 GiB vs 10 GiB, and why GQA exists |
| [`verify.py`](verify.py) | — | asserts all of the above against the printed book |

Every file runs on its own and prints an explanation as it goes:

```bash
python part2_transformer/ch08_self_attention.py
```

## Suggested order

Read the chapter first, then run its file, then change a number and run it again. The third
step is the one that teaches. Some things worth breaking on purpose:

- in `ch08`, delete the `/ np.sqrt(d_k)` and watch the attention weights collapse onto one word;
- in `ch14`, change `k=1` to `k=0` and watch every position lose the right to see itself;
- in `ch11`, drop `gamma` and `beta` and see how much of layer norm is actually learned (very little);
- in `ch03`, add the missing sentence to the corpus and watch an infinite perplexity become finite.

## Licence

MIT. Use it in your teaching, your notes, or your own book.
