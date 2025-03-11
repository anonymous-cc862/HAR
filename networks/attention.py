import flax.linen as nn
import jax.numpy as jnp
from typing import Type


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(nn.Module):
    # inner Q, K, V has dim = embed_dim/num_heads
    # if embed_dim != input_dim, it gives cross attention
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    @nn.compact
    def __call__(self, x, mask=None):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        batch_size, seq_length, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        if mask is not None:
            mask = expand_mask(mask)
        qkv = nn.Dense(3 * self.embed_dim,
                       kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                       bias_init=nn.initializers.zeros  # Bias init with zeros
                       )(x)  # qkv_proj

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)  # split the embed_dim = head*dims
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = nn.Dense(self.embed_dim,
                     kernel_init=nn.initializers.xavier_uniform(),
                     bias_init=nn.initializers.zeros)(values)  # output projection

        return o, attention


class EncoderBlock(nn.Module):
    input_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        # Attention part
        attn_out, _ = MultiHeadAttention(embed_dim=self.input_dim,
                                         num_heads=self.num_heads)(x, mask=mask)
        x = x + nn.Dropout(self.dropout_rate)(attn_out, deterministic=not training)
        x = nn.LayerNorm()(x)

        # MLP part
        linear_out = nn.Dense(self.mlp_dim)(x)
        linear_out = nn.Dropout(self.dropout_rate)(linear_out, deterministic=not training)
        linear_out = nn.relu(linear_out)
        linear_out = nn.Dense(self.input_dim)(linear_out)

        # add residual
        x = x + nn.Dropout(self.dropout_rate)(linear_out, deterministic=not training)
        x = nn.LayerNorm()(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        for _ in range(self.num_layers):
            x = EncoderBlock(input_dim=self.input_dim,
                             num_heads=self.num_heads,
                             mlp_dim=self.mlp_dim,
                             dropout_rate=self.dropout_rate)(x, mask=mask, training=training)
        return x


class TransformerEmbedding(nn.Module):
    hidden_dim: int
    time_embedding: Type[nn.Module]
    time_net: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,  # s = [obs, cond] if it's conditional  # (B, columns)
                 time: jnp.ndarray,
                 training: bool = False):

        batch_size, num_columns = x.shape

        t_ff = self.time_embedding()(time)
        time_suffix = self.time_net()(t_ff, training=training)  # (B, time_dim)

        # treat columns as sequence length in transformer
        # add CLS token (B, columns) -> (B, 1 + columns)
        x = jnp.concatenate([jnp.zeros((batch_size, 1)), x], axis=-1)
        # add time embeddings (B, 1 + columns, 1) -> (B, 1 + columns, 1 + time_dim)
        x = jnp.concatenate([jnp.expand_dims(x, axis=-1),
                             time_suffix[:, jnp.newaxis, :].repeat(num_columns + 1, axis=1)], axis=-1)
        # (B, 1 + columns, 1 + time_dim) -> (B, 1 + columns, D)
        x = nn.Dense(self.hidden_dim)(x)

        x = TransformerEncoder(num_layers=2,
                               input_dim=self.hidden_dim,
                               mlp_dim=64,
                               num_heads=2,
                               dropout_rate=0.1)(x, training=training)

        return x[:, 0, :]  # fetch the embedding of the CLS token





if __name__ == '__main__':
    import jax
    import jax.random as random

    main_rng = jax.random.PRNGKey(1)
    main_rng, x_rng = random.split(main_rng)
    xx = random.normal(x_rng, (3, 16, 128))
    # Create attention
    mh_attn = MultiHeadAttention(embed_dim=128, num_heads=4)
    # Initialize parameters of attention with random key and inputs
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = mh_attn.init(init_rng, xx)
    # Apply attention with parameters on the inputs
    out, attn = mh_attn.apply(params, xx)
    print('Out', out.shape, 'Attention', attn.shape)

    f = EncoderBlock(input_dim=128, mlp_dim=256, num_heads=4, dropout_rate=0.1)
    params = f.init({'params': init_rng, 'dropout': dropout_init_rng}, xx, training=True)
    y = f.apply(params, xx, training=True, rngs={'dropout': random.key(2)})

    f = TransformerEncoder(num_layers=5, input_dim=128, mlp_dim=256, num_heads=4, dropout_rate=0.1)
    params = f.init({'params': init_rng, 'dropout': dropout_init_rng}, xx, training=True)
    y = f.apply(params, xx, training=True, rngs={'dropout': random.key(2)})


    from diffusion_feature import FourierFeatures, mish
    from networks.mlp import MLP
    from functools import partial

    time_dim = 16

    time_embedding = partial(FourierFeatures,
                             output_size=time_dim,
                             learnable=False)

    time_net = partial(MLP,
                       hidden_dims=(32, 32),
                       activations=nn.leaky_relu,
                       activate_final=False)

    f = TransformerEmbedding(hidden_dim=12,  # should be the multiple of heads
                             time_embedding=time_embedding,
                             time_net=time_net)

    xx = random.normal(x_rng, (64, 8))
    params = f.init({'params': init_rng, 'dropout': dropout_init_rng},
                    jnp.ones((64, 8)),
                    jnp.ones((64, 1)),
                    training=True)
    y = f.apply(params, xx, jnp.ones((64, 1)), training=True, rngs={'dropout': random.key(2)})





