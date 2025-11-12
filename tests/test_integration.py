import pytest
import numpy as np
from vecrypt.orthogonal import orthogonal_from_seed
from vecrypt.encrypt import encrypt_embedding, decrypt_embedding
from vecrypt.secret_share import share_matrix


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_with_secret_sharing(self):
        """Test complete workflow: generate matrix, share it, encrypt vectors."""
        seed = b"integration_test"[:32].ljust(32, b"0")
        dim = 32
        num_parties = 3
        
        # Generate orthogonal matrix
        orthogonal_matrix = orthogonal_from_seed(seed, dim=dim)
        
        # Share the matrix
        matrix_shares = share_matrix(orthogonal_matrix, num_parties=num_parties)
        
        # Create embeddings
        embeddings = [
            np.random.randn(dim).astype(np.float64) 
            for _ in range(5)
        ]
        
        # Encrypt embeddings using direct matrix
        encrypted_direct = [
            encrypt_embedding(emb, seed) 
            for emb in embeddings
        ]
        
        # Encrypt embeddings using shares (simulating multi-party computation)
        encrypted_shared = []
        for emb in embeddings:
            # Each party computes its share of the encrypted vector
            party_results = [
                (emb.reshape(1, -1) @ share).flatten() 
                for share in matrix_shares
            ]
            # Sum the shares to get the encrypted vector
            encrypted_shared.append(sum(party_results))
        
        # Results should match
        for ed, es in zip(encrypted_direct, encrypted_shared):
            np.testing.assert_allclose(ed, es, atol=1e-10)
        
        # Verify decryption works
        for original, encrypted in zip(embeddings, encrypted_direct):
            decrypted = decrypt_embedding(encrypted, seed)
            np.testing.assert_allclose(decrypted, original, atol=1e-10)

    def test_similarity_search_preservation(self):
        """Test that similarity search results are preserved after encryption."""
        seed = b"similarity_test"[:32].ljust(32, b"0")
        dim = 64
        
        # Create a query and multiple candidates
        query = np.random.randn(dim).astype(np.float64)
        candidates = [
            np.random.randn(dim).astype(np.float64) 
            for _ in range(10)
        ]
        
        # Compute original similarities
        original_similarities = np.array([
            np.dot(query, cand) for cand in candidates
        ])
        original_top_k_indices = np.argsort(original_similarities)[::-1][:5]
        
        # Encrypt all vectors
        encrypted_query = encrypt_embedding(query, seed)
        encrypted_candidates = [
            encrypt_embedding(cand, seed) 
            for cand in candidates
        ]
        
        # Compute encrypted similarities
        encrypted_similarities = np.array([
            np.dot(encrypted_query, enc_cand) 
            for enc_cand in encrypted_candidates
        ])
        encrypted_top_k_indices = np.argsort(encrypted_similarities)[::-1][:5]
        
        # Top-K results should be identical
        np.testing.assert_array_equal(original_top_k_indices, encrypted_top_k_indices)
        
        # Similarity scores should match exactly
        np.testing.assert_allclose(
            original_similarities[original_top_k_indices],
            encrypted_similarities[encrypted_top_k_indices],
            atol=1e-10
        )

    def test_secret_shared_encryption_preserves_similarity(self):
        """Test that encryption with secret-shared matrix preserves similarity."""
        seed = b"shared_similarity"[:32].ljust(32, b"0")
        dim = 32
        num_parties = 4
        
        # Generate and share orthogonal matrix
        orthogonal_matrix = orthogonal_from_seed(seed, dim=dim)
        matrix_shares = share_matrix(orthogonal_matrix, num_parties=num_parties)
        
        # Create vectors
        v1 = np.random.randn(dim).astype(np.float64)
        v2 = np.random.randn(dim).astype(np.float64)
        
        # Compute original similarity
        original_similarity = np.dot(v1, v2)
        
        # Encrypt using shares (multi-party computation)
        # Each party computes share @ v
        v1_encrypted_shares = [
            (v1.reshape(1, -1) @ share).flatten() 
            for share in matrix_shares
        ]
        v2_encrypted_shares = [
            (v2.reshape(1, -1) @ share).flatten() 
            for share in matrix_shares
        ]
        
        # Sum shares to get encrypted vectors
        v1_encrypted = sum(v1_encrypted_shares)
        v2_encrypted = sum(v2_encrypted_shares)
        
        # Compute encrypted similarity
        encrypted_similarity = np.dot(v1_encrypted, v2_encrypted)
        
        # Similarities should match
        np.testing.assert_allclose(original_similarity, encrypted_similarity, atol=1e-10)
        
        # Also verify against direct encryption
        v1_direct = encrypt_embedding(v1, seed)
        v2_direct = encrypt_embedding(v2, seed)
        direct_similarity = np.dot(v1_direct, v2_direct)
        
        np.testing.assert_allclose(original_similarity, direct_similarity, atol=1e-10)
        np.testing.assert_allclose(encrypted_similarity, direct_similarity, atol=1e-10)

    def test_batch_encryption_decryption(self):
        """Test encrypting and decrypting multiple embeddings."""
        seed = b"batch_test"[:32].ljust(32, b"0")
        dim = 64
        num_embeddings = 100
        
        # Generate batch of embeddings
        embeddings = [
            np.random.randn(dim).astype(np.float64) 
            for _ in range(num_embeddings)
        ]
        
        # Encrypt all
        encrypted = [
            encrypt_embedding(emb, seed) 
            for emb in embeddings
        ]
        
        # Decrypt all
        decrypted = [
            decrypt_embedding(enc, seed) 
            for enc in encrypted
        ]
        
        # Verify all roundtrips
        for original, dec in zip(embeddings, decrypted):
            np.testing.assert_allclose(original, dec, atol=1e-10)

    def test_consistency_across_multiple_runs(self):
        """Test that same seed produces consistent results across runs."""
        seed = b"consistency_test"[:32].ljust(32, b"0")
        dim = 32
        embedding = np.random.randn(dim).astype(np.float64)
        
        # Encrypt multiple times with same seed
        encrypted1 = encrypt_embedding(embedding, seed)
        encrypted2 = encrypt_embedding(embedding, seed)
        encrypted3 = encrypt_embedding(embedding, seed)
        
        # All should be identical
        np.testing.assert_allclose(encrypted1, encrypted2, atol=1e-10)
        np.testing.assert_allclose(encrypted2, encrypted3, atol=1e-10)
        
        # Decrypt all
        decrypted1 = decrypt_embedding(encrypted1, seed)
        decrypted2 = decrypt_embedding(encrypted2, seed)
        decrypted3 = decrypt_embedding(encrypted3, seed)
        
        # All should recover original
        for dec in [decrypted1, decrypted2, decrypted3]:
            np.testing.assert_allclose(dec, embedding, atol=1e-10)

