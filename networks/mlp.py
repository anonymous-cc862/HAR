
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Type
from networks.initialization import default_init


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:  # hidden layers
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


# class TimeMLP(nn.Module):
#     mlp: Type[nn.Module]
#     time_embedding: Type[nn.Module]
#     time_processor: Type[nn.Module]
#     """
#     if cond_embedding:
#         treat the last dim as the conditional tag
#     """
#     @nn.compact
#     def __call__(self,
#                  x: jnp.ndarray,  # s = [obs, cond] if it's conditional
#                  time: jnp.ndarray,
#                  training: bool = False):
#         t_ff = self.time_embedding()(time)
#         time_suffix = self.time_processor()(t_ff, training=training)
#         input_with_time = jnp.concatenate([x, time_suffix], axis=-1)
#         return self.mlp()(input_with_time, training=training)
#

