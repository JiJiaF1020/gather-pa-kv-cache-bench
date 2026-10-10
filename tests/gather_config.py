"""Payload layout and runtime defaults for the GatherPaKvCache test tools."""

DTYPE_K = 'fp8_e4m3fn'
DTYPE_V = 'bf16'
DTYPE_K_BYTES = 1
DTYPE_V_BYTES = 2
NUM_HEADS = 1
HEAD_SIZE_K = 512
HEAD_SIZE_V = 72
BLOCK_SIZE = 128


def bytes_per_token(num_heads: int, head_size_k: int, head_size_v: int) -> int:
    """Effective bytes gathered per token for the packed MLA row layout."""
    return num_heads * (head_size_k * DTYPE_K_BYTES + head_size_v * DTYPE_V_BYTES)


# Default C8 packed MLA row: 512B FP8 latent + 128B RoPE + 16B scale = 656B.
BYTES_PER_TOKEN = bytes_per_token(NUM_HEADS, HEAD_SIZE_K, HEAD_SIZE_V)
DEFAULT_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
LEGACY_VISIBLE_ENV = 'A5_TEST_VISIBLE_DEVICES'
RESULTS_SUBDIR = 'results/gather_pa_kv_cache'
MULTIPROCESS_TEST = 'a5_gather_pa_kv_cache_multiprocess'
SINGLECARD_TEST = 'a5_gather_pa_kv_cache_singlecard'
