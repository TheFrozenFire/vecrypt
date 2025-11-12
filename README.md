# Vecrypt

A Python library for encrypting embedding vectors using orthogonal transformations while preserving similarity search capabilities. Vecrypt enables secure storage and retrieval of embeddings in vector databases without compromising search accuracy.

## Features

- 🔐 **Zero-loss encryption**: Preserves dot products and cosine similarity exactly
- 🔑 **Seed-based encryption**: Deterministic encryption using a 256-bit secret seed
- 🔄 **Reversible**: Decrypt embeddings back to original space with the correct seed
- 👥 **Secret sharing support**: Distribute encryption keys across multiple parties using additive secret sharing
- 📊 **Dimension-agnostic**: Works with embeddings of any dimension
- ⚡ **Efficient**: Built on NumPy for high-performance matrix operations

## Installation

Vecrypt requires Python 3.10 or higher.

### Using uv (recommended)

```bash
uv add vecrypt
```

### Using pip

```bash
pip install vecrypt
```

## Quick Start

> **Important**: Always use `secrets.token_bytes(32)` to generate seeds securely. Never use hardcoded strings or predictable values for production use.

### Basic Encryption and Decryption

```python
import secrets
import numpy as np
from vecrypt import encrypt_embedding, decrypt_embedding

# Generate a cryptographically secure 256-bit (32-byte) secret seed
seed = secrets.token_bytes(32)

# Create an embedding vector (any dimension)
embedding = np.random.randn(1024).astype(np.float64)

# Encrypt the embedding
encrypted = encrypt_embedding(embedding, seed)

# Decrypt back to original
decrypted = decrypt_embedding(encrypted, seed)

# Verify roundtrip
assert np.allclose(embedding, decrypted)
```

### Preserving Similarity Search

```python
import secrets
import numpy as np
from vecrypt import encrypt_embedding

# Generate a secure seed (store this securely for decryption!)
seed = secrets.token_bytes(32)

# Create query and candidate embeddings
query = np.random.randn(1024).astype(np.float64)
candidates = [np.random.randn(1024).astype(np.float64) for _ in range(10)]

# Compute original similarities
original_similarities = [np.dot(query, cand) for cand in candidates]
original_ranking = np.argsort(original_similarities)[::-1]

# Encrypt all vectors
encrypted_query = encrypt_embedding(query, seed)
encrypted_candidates = [encrypt_embedding(cand, seed) for cand in candidates]

# Compute encrypted similarities
encrypted_similarities = [np.dot(encrypted_query, enc_cand) 
                          for enc_cand in encrypted_candidates]
encrypted_ranking = np.argsort(encrypted_similarities)[::-1]

# Rankings are identical!
assert np.array_equal(original_ranking, encrypted_ranking)
```

### Secret Sharing for Multi-Party Computation

```python
import secrets
import numpy as np
from vecrypt.orthogonal import orthogonal_from_seed
from vecrypt.secret_share import share_matrix

# Generate a secure seed
seed = secrets.token_bytes(32)
dim = 1024
num_parties = 3

# Generate orthogonal matrix
orthogonal_matrix = orthogonal_from_seed(seed, dim=dim)

# Secret-share the matrix among parties
matrix_shares = share_matrix(orthogonal_matrix, num_parties=num_parties)

# Each party now holds a share of the matrix
# The original matrix can be discarded
# No single party knows the complete matrix

# To encrypt an embedding using shares:
embedding = np.random.randn(dim).astype(np.float64)

# Each party computes: share @ embedding
party_results = [(embedding.reshape(1, -1) @ share).flatten() 
                 for share in matrix_shares]

# Sum the shares to get the encrypted vector
encrypted = sum(party_results)

# The encrypted vector preserves similarity properties
# without any party learning the original embedding or matrix
```

## API Reference

### `encrypt_embedding(embedding, seed)`

Encrypts an embedding vector using an orthogonal transformation.

**Parameters:**
- `embedding` (np.ndarray): The embedding vector to encrypt (1D array of any dimension)
- `seed` (bytes): 32-byte (256-bit) secret key (use `secrets.token_bytes(32)` to generate securely)

**Returns:**
- `np.ndarray`: Encrypted embedding vector (same dimension as input)

**Example:**
```python
encrypted = encrypt_embedding(embedding, seed)
```

### `decrypt_embedding(encrypted_embedding, seed)`

Decrypts an encrypted embedding back to the original space.

**Parameters:**
- `encrypted_embedding` (np.ndarray): The encrypted embedding vector
- `seed` (bytes): 32-byte secret key (must match the one used for encryption; use `secrets.token_bytes(32)` to generate securely)

**Returns:**
- `np.ndarray`: Decrypted embedding vector

**Example:**
```python
original = decrypt_embedding(encrypted, seed)
```

### `orthogonal_from_seed(seed, dim)`

Generates a deterministic orthogonal matrix from a seed.

**Parameters:**
- `seed` (bytes): 32-byte (256-bit) secret seed (use `secrets.token_bytes(32)` to generate securely)
- `dim` (int): Dimension of the embedding vectors

**Returns:**
- `np.ndarray`: Orthogonal matrix of shape (dim, dim)

**Example:**
```python
matrix = orthogonal_from_seed(seed, dim=1024)
```

### `share_matrix(matrix, num_parties, seed=None)`

Secret-shares a matrix among multiple parties using additive secret sharing.

**Parameters:**
- `matrix` (np.ndarray): The matrix to share
- `num_parties` (int): Number of parties to split the secret among (must be ≥ 2)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `list[np.ndarray]`: List of share matrices, one per party

**Example:**
```python
shares = share_matrix(matrix, num_parties=3)
# shares[0] + shares[1] + shares[2] == matrix
```

## How It Works

Vecrypt uses orthogonal matrix transformations to encrypt embeddings. The key insight is that orthogonal transformations preserve dot products and angles between vectors, which are exactly what similarity search relies on.

1. **Matrix Generation**: A random orthogonal matrix is generated from a 256-bit seed using QR decomposition
2. **Encryption**: The embedding is multiplied by the orthogonal matrix: `encrypted = embedding @ matrix`
3. **Decryption**: The encrypted vector is multiplied by the transpose: `original = encrypted @ matrix.T`
4. **Similarity Preservation**: Because `(A @ M) @ (B @ M).T = A @ B.T`, dot products are preserved exactly

The orthogonal matrix can be secret-shared among multiple parties, enabling secure multi-party computation where no single party learns the encryption key or the original embeddings.

For detailed mathematical background, see [EXPLANATION.md](EXPLANATION.md).

## Testing

Run the test suite with:

```bash
uv run pytest tests/ -v
```

Or with pip:

```bash
pytest tests/ -v
```

The test suite includes:
- Unit tests for each module
- Integration tests for full workflows
- Tests verifying similarity preservation
- Tests for secret sharing functionality

## Requirements

- Python ≥ 3.10
- NumPy

## License

MIT License

## Author

Justin Martin - justin@thefrozenfire.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

