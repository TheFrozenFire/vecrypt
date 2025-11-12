import numpy as np

def share_matrix(
    matrix: np.ndarray,
    num_parties: int,
    seed: int | None = None
) -> list[np.ndarray]:
    """
    Additively secret-share a matrix among num_parties parties.
    
    Each party receives a share such that the sum of all shares equals the
    original matrix. No party learns the full matrix.
    
    Args:
        matrix: The matrix to share.
        num_parties: Number of parties to split the secret among.
        seed: Optional random seed for reproducibility.
    
    Returns:
        A list of numpy arrays (one per party), each of shape (d, d),
        representing that party's additive share of the matrix.
    """
    if num_parties < 2:
        raise ValueError("Number of parties must be at least 2.")

    # Set seed for reproducibility across all parties
    rng = np.random.default_rng(seed)
    
    # Step 3: Additively secret-share Q among num_parties
    # Generate (num_parties - 1) random share matrices
    shares = [rng.standard_normal(size=matrix.shape) for _ in range(num_parties - 1)]
    
    # The last share is Q minus the sum of all previous shares
    last_share = matrix.copy()
    for share in shares:
        last_share -= share
    
    # Append the final computed share
    shares.append(last_share)
    
    reconstructed = sum(shares)
    if not np.allclose(reconstructed, matrix, atol=1e-10):
        raise RuntimeError("Secret sharing reconstruction failed.")
    
    return shares