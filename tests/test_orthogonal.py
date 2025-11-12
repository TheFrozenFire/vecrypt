import pytest
import numpy as np
from vecrypt.orthogonal import orthogonal_from_seed


class TestOrthogonalFromSeed:
    """Tests for orthogonal matrix generation from seed."""

    def test_seed_length_validation(self):
        """Test that seed must be exactly 32 bytes."""
        with pytest.raises(ValueError, match="Seed must be exactly 32 bytes"):
            orthogonal_from_seed(b"short", dim=10)
        
        with pytest.raises(ValueError, match="Seed must be exactly 32 bytes"):
            orthogonal_from_seed(b"x" * 31, dim=10)
        
        with pytest.raises(ValueError, match="Seed must be exactly 32 bytes"):
            orthogonal_from_seed(b"x" * 33, dim=10)

    def test_matrix_shape(self):
        """Test that generated matrix has correct shape."""
        seed = b"x" * 32
        dim = 64
        Q = orthogonal_from_seed(seed, dim=dim)
        
        assert Q.shape == (dim, dim)
        assert Q.dtype == np.float64

    def test_orthogonality_property(self):
        """Test that generated matrix is orthogonal (Q^T @ Q = I)."""
        seed = b"x" * 32
        dim = 64
        Q = orthogonal_from_seed(seed, dim=dim)
        
        # Q^T @ Q should equal identity matrix
        identity = Q.T @ Q
        expected_identity = np.eye(dim)
        
        np.testing.assert_allclose(identity, expected_identity, atol=1e-10)

    def test_determinism(self):
        """Test that same seed produces same matrix."""
        seed = b"y" * 32
        dim = 32
        
        Q1 = orthogonal_from_seed(seed, dim=dim)
        Q2 = orthogonal_from_seed(seed, dim=dim)
        
        np.testing.assert_array_equal(Q1, Q2)

    def test_different_seeds_produce_different_matrices(self):
        """Test that different seeds produce different matrices."""
        seed1 = b"a" * 32
        seed2 = b"b" * 32
        dim = 32
        
        Q1 = orthogonal_from_seed(seed1, dim=dim)
        Q2 = orthogonal_from_seed(seed2, dim=dim)
        
        # Matrices should be different
        assert not np.allclose(Q1, Q2, atol=1e-10)
        
        # But both should still be orthogonal
        np.testing.assert_allclose(Q1.T @ Q1, np.eye(dim), atol=1e-10)
        np.testing.assert_allclose(Q2.T @ Q2, np.eye(dim), atol=1e-10)

    def test_different_dimensions(self):
        """Test that function works with different dimensions."""
        seed = b"z" * 32
        
        for dim in [16, 32, 64, 128, 256]:
            Q = orthogonal_from_seed(seed, dim=dim)
            assert Q.shape == (dim, dim)
            np.testing.assert_allclose(Q.T @ Q, np.eye(dim), atol=1e-10)

    def test_inverse_is_transpose(self):
        """Test that Q^(-1) = Q^T for orthogonal matrices."""
        seed = b"w" * 32
        dim = 64
        Q = orthogonal_from_seed(seed, dim=dim)
        
        # Q @ Q^T should also equal identity
        identity = Q @ Q.T
        expected_identity = np.eye(dim)
        
        np.testing.assert_allclose(identity, expected_identity, atol=1e-10)

    def test_preserves_vector_lengths(self):
        """Test that orthogonal transformation preserves vector lengths."""
        seed = b"v" * 32
        dim = 64
        Q = orthogonal_from_seed(seed, dim=dim)
        
        # Create a random vector
        v = np.random.randn(dim)
        v_normalized = v / np.linalg.norm(v)
        
        # Transform the vector
        v_transformed = Q @ v_normalized
        
        # Length should be preserved
        original_length = np.linalg.norm(v_normalized)
        transformed_length = np.linalg.norm(v_transformed)
        
        np.testing.assert_allclose(original_length, transformed_length, atol=1e-10)

