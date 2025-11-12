from typing import List
import numpy as np
from .orthogonal import orthogonal_from_seed

def encrypt_embedding(
    embedding: np.ndarray,
    seed: bytes,
) -> np.ndarray:
    """
    Encrypt an embedding using a seed
    
    Args:
        embedding: The embedding to encrypt.
        seed: 32-byte (256-bit) secret key. Use secrets.token_bytes(32) to generate securely.
    
    Returns:
        Encrypted embedding vector (same dimension as input)
    """
    # Generate or load R_K
    R_K = orthogonal_from_seed(seed, embedding.shape[0])

    # Apply rotation: e' = e @ R_K
    encrypted = embedding @ R_K

    return encrypted


def decrypt_embedding(
    encrypted_embedding: np.ndarray,
    seed: bytes,
) -> np.ndarray:
    """
    Decrypt an encrypted embedding back to the original space.
    
    Args:
        encrypted_embedding: The encrypted embedding vector
        seed: 32-byte (256-bit) secret key. Use secrets.token_bytes(32) to generate securely.
            Must match the seed used for encryption.
    
    Returns:
        Decrypted embedding vector
    
    Note: Only works if same seed was used for encryption.
    """
    R_K = orthogonal_from_seed(seed, encrypted_embedding.shape[0])

    # Reverse: e = e' @ R_K^T  (since R_K^{-1} = R_K^T)
    original = encrypted_embedding @ R_K.T

    return original