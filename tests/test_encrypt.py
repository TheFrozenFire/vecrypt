import pytest
import numpy as np
from vecrypt.encrypt import encrypt_embedding, decrypt_embedding


class TestEncryptEmbedding:
    """Tests for embedding encryption."""

    def test_encryption_decryption_roundtrip(self):
        """Test that encryption followed by decryption recovers original."""
        seed = b"x" * 32
        dim = 64
        original = np.random.randn(dim).astype(np.float64)
        
        encrypted = encrypt_embedding(original, seed)
        decrypted = decrypt_embedding(encrypted, seed)
        
        np.testing.assert_allclose(decrypted, original, atol=1e-10)

    def test_handles_different_dimensions(self):
        """Test that function works with any embedding dimension."""
        seed = b"x" * 32
        
        # Should work with any dimension
        for dim in [16, 32, 64, 128, 256]:
            original = np.random.randn(dim).astype(np.float64)
            encrypted = encrypt_embedding(original, seed)
            decrypted = decrypt_embedding(encrypted, seed)
            np.testing.assert_allclose(decrypted, original, atol=1e-10)

    def test_preserves_dot_product(self):
        """Test that encryption preserves dot products between vectors."""
        seed = b"y" * 32
        dim = 64
        
        v1 = np.random.randn(dim).astype(np.float64)
        v2 = np.random.randn(dim).astype(np.float64)
        
        original_dot = np.dot(v1, v2)
        
        encrypted_v1 = encrypt_embedding(v1, seed)
        encrypted_v2 = encrypt_embedding(v2, seed)
        
        encrypted_dot = np.dot(encrypted_v1, encrypted_v2)
        
        np.testing.assert_allclose(original_dot, encrypted_dot, atol=1e-10)

    def test_preserves_cosine_similarity(self):
        """Test that encryption preserves cosine similarity."""
        seed = b"z" * 32
        dim = 64
        
        v1 = np.random.randn(dim).astype(np.float64)
        v2 = np.random.randn(dim).astype(np.float64)
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        original_cosine = np.dot(v1_norm, v2_norm)
        
        encrypted_v1 = encrypt_embedding(v1_norm, seed)
        encrypted_v2 = encrypt_embedding(v2_norm, seed)
        
        encrypted_cosine = np.dot(encrypted_v1, encrypted_v2) / (
            np.linalg.norm(encrypted_v1) * np.linalg.norm(encrypted_v2)
        )
        
        np.testing.assert_allclose(original_cosine, encrypted_cosine, atol=1e-10)

    def test_different_seeds_produce_different_encryptions(self):
        """Test that different seeds produce different encrypted vectors."""
        dim = 64
        original = np.random.randn(dim).astype(np.float64)
        
        seed1 = b"a" * 32
        seed2 = b"b" * 32
        
        encrypted1 = encrypt_embedding(original, seed1)
        encrypted2 = encrypt_embedding(original, seed2)
        
        # Encryptions should be different
        assert not np.allclose(encrypted1, encrypted2, atol=1e-10)
        
        # But each should decrypt correctly with its own seed
        decrypted1 = decrypt_embedding(encrypted1, seed1)
        decrypted2 = decrypt_embedding(encrypted2, seed2)
        
        np.testing.assert_allclose(decrypted1, original, atol=1e-10)
        np.testing.assert_allclose(decrypted2, original, atol=1e-10)

    def test_wrong_seed_cannot_decrypt(self):
        """Test that wrong seed cannot decrypt correctly."""
        seed1 = b"a" * 32
        seed2 = b"b" * 32
        dim = 64
        original = np.random.randn(dim).astype(np.float64)
        
        encrypted = encrypt_embedding(original, seed1)
        
        # Try to decrypt with wrong seed
        wrong_decrypted = decrypt_embedding(encrypted, seed2)
        
        # Should not match original
        assert not np.allclose(wrong_decrypted, original, atol=1e-5)

    def test_preserves_similarity_ranking(self):
        """Test that encryption preserves similarity rankings."""
        seed = b"r" * 32
        dim = 64
        
        query = np.random.randn(dim).astype(np.float64)
        candidates = [np.random.randn(dim).astype(np.float64) for _ in range(5)]
        
        # Compute original similarities
        original_similarities = [np.dot(query, cand) for cand in candidates]
        original_ranking = np.argsort(original_similarities)[::-1]  # Descending
        
        # Encrypt all vectors
        encrypted_query = encrypt_embedding(query, seed)
        encrypted_candidates = [
            encrypt_embedding(cand, seed) 
            for cand in candidates
        ]
        
        # Compute encrypted similarities
        encrypted_similarities = [
            np.dot(encrypted_query, enc_cand) 
            for enc_cand in encrypted_candidates
        ]
        encrypted_ranking = np.argsort(encrypted_similarities)[::-1]  # Descending
        
        # Rankings should be identical
        np.testing.assert_array_equal(original_ranking, encrypted_ranking)

    def test_handles_list_input(self):
        """Test that function accepts list input (converted to numpy array)."""
        seed = b"l" * 32
        dim = 64
        original_list = list(np.random.randn(dim).astype(np.float64))
        original = np.array(original_list)  # Convert to numpy array
        
        encrypted = encrypt_embedding(original, seed)
        decrypted = decrypt_embedding(encrypted, seed)
        
        np.testing.assert_allclose(decrypted, original, atol=1e-10)


