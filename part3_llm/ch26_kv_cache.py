"""Chapter 26: the KV cache, and why long context costs so much memory.

Generating token t recomputes K and V for every earlier token, every time.
Cache them instead and each new token is O(1) work rather than O(t). The
bill arrives as memory, and this file prints that bill.

    python part3_llm/ch26_kv_cache.py
"""


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per_number=2):
    """Bytes held by the KV cache.

    Times 2 because we store BOTH a key and a value per position.
    bytes_per_number = 2 is fp16, the usual serving choice.
    """
    return 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_number


def gib(n_bytes):
    return n_bytes / 1024 ** 3


if __name__ == "__main__":
    # a 70B-class model, the shape used in the book
    LAYERS, HEADS, D_HEAD = 80, 64, 128
    SEQ = 32768

    mha = kv_cache_bytes(LAYERS, HEADS, D_HEAD, SEQ)
    print("Multi-head attention, 64 KV heads")
    print("  %d layers x %d heads x %d dims x %d tokens x 2 (K and V) x 2 bytes"
          % (LAYERS, HEADS, D_HEAD, SEQ))
    print("  = %.0f GiB  for ONE user's context" % gib(mha))

    print("\nGrouped-query attention, 8 KV heads (the same model, fewer KV heads)")
    gqa = kv_cache_bytes(LAYERS, 8, D_HEAD, SEQ)
    print("  = %.0f GiB" % gib(gqa))

    print("\nratio = %.0f, which is exactly H/G = %d/%d." % (mha / gqa, HEADS, 8))
    print("That ratio IS the entire argument for GQA. Nothing about")
    print("quality: it is 80 GiB of memory per user versus 10.")

    print("\nhow the cache grows with context length (GQA-8):")
    for seq in (1024, 4096, 16384, 32768, 131072):
        print("  %7d tokens : %6.1f GiB" % (seq, gib(kv_cache_bytes(LAYERS, 8, D_HEAD, seq))))
    print("\nLinear in sequence length, and that line is why 'long context'")
    print("is a systems problem before it is a modelling problem.")
