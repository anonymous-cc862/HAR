
import flax.linen as nn
import jax
import jax.numpy as jnp


class CodeBook(nn.Module):
    embedding_dim: int
    num_codes: int

    def setup(self):
        self.codebook = self.param('codebook', nn.initializers.lecun_uniform(),
                                   (self.num_codes, self.embedding_dim))  # (K, D)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        l2_sum = jax.vmap(lambda x_: ((self.codebook - x_) ** 2).sum(axis=-1))(x)  # (B, z_dim) -> (B, num_ways)
        codes = l2_sum.argmin(axis=-1)  # (B,)
        code_vec = self.codebook[codes]  # (B, embedding_dim)

        return code_vec, codes, l2_sum
