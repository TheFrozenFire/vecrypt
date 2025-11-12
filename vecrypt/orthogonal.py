import numpy as np

def orthogonal_from_seed(seed: bytes, dim: int) -> np.ndarray:
    """
    Generate a deterministic orthogonal matrix R_K from a 256-bit seed.
    
    Args:
        seed: 32-byte (256-bit) cryptographically secure seed.
            Use secrets.token_bytes(32) to generate securely.
        dim: Embedding dimension
    
    Returns:
        R_K: (dim, dim) orthogonal matrix, float64
    """
    if len(seed) != 32:
        raise ValueError("Seed must be exactly 32 bytes (256 bits)")

    # Use SeedSequence for cryptographically sound PRNG initialization
    ss = np.random.SeedSequence(entropy=list[int](seed))
    rng = np.random.default_rng(ss)

    # Generate random matrix
    A = rng.standard_normal((dim, dim), dtype=np.float64)

    # QR decomposition → Q is orthogonal
    Q, _ = np.linalg.qr(A)

    return Q