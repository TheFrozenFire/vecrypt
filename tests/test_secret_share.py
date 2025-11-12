import pytest
import numpy as np
from vecrypt.secret_share import share_matrix


class TestShareMatrix:
    """Tests for matrix secret sharing."""

    def test_num_parties_validation(self):
        """Test that num_parties must be at least 2."""
        matrix = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Number of parties must be at least 2"):
            share_matrix(matrix, num_parties=1)
        
        with pytest.raises(ValueError, match="Number of parties must be at least 2"):
            share_matrix(matrix, num_parties=0)
        
        with pytest.raises(ValueError, match="Number of parties must be at least 2"):
            share_matrix(matrix, num_parties=-1)

    def test_shares_sum_to_original(self):
        """Test that all shares sum to the original matrix."""
        matrix = np.random.randn(16, 16)
        num_parties = 3
        
        shares = share_matrix(matrix, num_parties=num_parties)
        
        reconstructed = sum(shares)
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-10)

    def test_correct_number_of_shares(self):
        """Test that correct number of shares is returned."""
        matrix = np.random.randn(8, 8)
        
        for num_parties in [2, 3, 5, 10]:
            shares = share_matrix(matrix, num_parties=num_parties)
            assert len(shares) == num_parties
            assert all(share.shape == matrix.shape for share in shares)

    def test_determinism_with_seed(self):
        """Test that same seed produces same shares."""
        matrix = np.random.randn(10, 10)
        num_parties = 4
        seed = 42
        
        shares1 = share_matrix(matrix, num_parties=num_parties, seed=seed)
        shares2 = share_matrix(matrix, num_parties=num_parties, seed=seed)
        
        for s1, s2 in zip(shares1, shares2):
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shares(self):
        """Test that different seeds produce different shares."""
        matrix = np.random.randn(10, 10)
        num_parties = 3
        
        shares1 = share_matrix(matrix, num_parties=num_parties, seed=1)
        shares2 = share_matrix(matrix, num_parties=num_parties, seed=2)
        
        # Shares should be different
        for s1, s2 in zip(shares1, shares2):
            assert not np.allclose(s1, s2, atol=1e-10)
        
        # But both should reconstruct correctly
        reconstructed1 = sum(shares1)
        reconstructed2 = sum(shares2)
        np.testing.assert_allclose(reconstructed1, matrix, atol=1e-10)
        np.testing.assert_allclose(reconstructed2, matrix, atol=1e-10)

    def test_individual_share_does_not_reveal_matrix(self):
        """Test that individual shares don't reveal the original matrix."""
        matrix = np.random.randn(8, 8)
        num_parties = 5
        
        shares = share_matrix(matrix, num_parties=num_parties)
        
        # Each share should be significantly different from the original
        for share in shares:
            assert not np.allclose(share, matrix, atol=1e-5)
            
            # Correlation should be low (not perfect match)
            correlation = np.corrcoef(share.flatten(), matrix.flatten())[0, 1]
            assert abs(correlation) < 0.9  # Should not be highly correlated

    def test_matrix_vector_multiplication_with_shares(self):
        """Test that matrix-vector multiplication works with shares."""
        # This tests the key property: (sum of shares) @ v = sum of (share @ v)
        matrix = np.random.randn(16, 16)
        vector = np.random.randn(16)
        num_parties = 4
        
        shares = share_matrix(matrix, num_parties=num_parties)
        
        # Direct multiplication
        direct_result = matrix @ vector
        
        # Share-wise multiplication and sum
        share_results = [share @ vector for share in shares]
        share_sum_result = sum(share_results)
        
        np.testing.assert_allclose(direct_result, share_sum_result, atol=1e-10)

    def test_orthogonal_matrix_sharing(self):
        """Test sharing an orthogonal matrix (as used in encryption)."""
        from vecrypt.orthogonal import orthogonal_from_seed
        
        seed = b"x" * 32
        dim = 32
        orthogonal_matrix = orthogonal_from_seed(seed, dim=dim)
        
        num_parties = 3
        shares = share_matrix(orthogonal_matrix, num_parties=num_parties)
        
        # Reconstruct and verify it's still orthogonal
        reconstructed = sum(shares)
        np.testing.assert_allclose(reconstructed, orthogonal_matrix, atol=1e-10)
        
        # Reconstructed matrix should still be orthogonal
        identity = reconstructed.T @ reconstructed
        expected_identity = np.eye(dim)
        np.testing.assert_allclose(identity, expected_identity, atol=1e-10)

    def test_encrypted_vector_with_shared_matrix(self):
        """Test that encryption works with secret-shared orthogonal matrix."""
        from vecrypt.orthogonal import orthogonal_from_seed
        
        seed = b"y" * 32
        dim = 32
        embedding = np.random.randn(dim).astype(np.float64)
        
        # Generate orthogonal matrix and share it
        orthogonal_matrix = orthogonal_from_seed(seed, dim=dim)
        num_parties = 3
        matrix_shares = share_matrix(orthogonal_matrix, num_parties=num_parties)
        
        # Encrypt using direct matrix
        direct_encrypted = (embedding.reshape(1, -1) @ orthogonal_matrix).flatten()
        
        # Encrypt using shares (each party computes share @ embedding, then sum)
        share_encrypted_parts = [
            (embedding.reshape(1, -1) @ share).flatten() 
            for share in matrix_shares
        ]
        share_encrypted = sum(share_encrypted_parts)
        
        # Results should match
        np.testing.assert_allclose(direct_encrypted, share_encrypted, atol=1e-10)

    def test_different_matrix_shapes(self):
        """Test sharing matrices of different shapes."""
        for shape in [(4, 4), (8, 8), (16, 16), (32, 32), (10, 20)]:
            matrix = np.random.randn(*shape)
            num_parties = 3
            
            shares = share_matrix(matrix, num_parties=num_parties)
            reconstructed = sum(shares)
            
            np.testing.assert_allclose(reconstructed, matrix, atol=1e-10)
            assert all(share.shape == shape for share in shares)

    def test_large_number_of_parties(self):
        """Test sharing with many parties."""
        matrix = np.random.randn(20, 20)
        num_parties = 20
        
        shares = share_matrix(matrix, num_parties=num_parties)
        reconstructed = sum(shares)
        
        assert len(shares) == num_parties
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-10)

