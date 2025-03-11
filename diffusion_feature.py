import numpy as np
import openml
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from networks.types import Optional, Sequence, Union, InfoDict, Tuple, Callable
from functools import partial
from networks.mlp import MLP
from networks.codebook import CodeBook
from networks.initialization import orthogonal_init
from networks.model import Model
from networks.types import Batch, MetaBatch, PRNGKey, Params
from diffusions.diffusion import DDPM, ddpm_sampler
from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule

EPS = 1e-6


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


@partial(jax.jit, static_argnames=('embed_t_dependent', 'alpha', 'p', 'no_diffusion'))
def jit_update_diffusion_model_with_embedding(noise_model: Model,
                                              embed_model: Model,
                                              codebook: Model,
                                              embed_t_dependent: bool,
                                              alpha: float,
                                              std_scale: np.ndarray,
                                              batch: Batch,
                                              rng: PRNGKey,
                                              T: int,
                                              alpha_hats: jnp.ndarray,
                                              p: float,  # col_select_ratio
                                              no_diffusion: bool
                                              ) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    batch_size, x_dim = batch.X.shape
    n_num_feature, n_cat_feature = len(batch.num_idx), len(batch.cate_idx)
    rng, t_key, noise_key, drop_key1, drop_key2, z_key, map_key1, map_key2 = jax.random.split(rng, 8)
    t = jax.random.randint(t_key, (batch_size, 1), 1, T + 1)  # t = 1, 2,...,T
    sigma = std_scale[jnp.newaxis, :].repeat(batch_size, axis=0)
    embed_t = t if embed_t_dependent else jnp.zeros(t.shape)

    eps_sample = jax.random.normal(noise_key, batch.X.shape)

    perm_idx = (jnp.arange(batch_size) - 1) % batch_size  # dislocation: [0, 1, 2,..., N] -> [N, 0, 1, 2,..., N-1]
    # p = 1/(n_cat_feature + EPS)
    A = jnp.sqrt(3 * p * (1 - p))  # variance preserving scale
    W1 = jax.random.uniform(map_key1, (n_num_feature, n_num_feature), minval=-A, maxval=A)
    W2 = jax.random.bernoulli(map_key2, p, (n_cat_feature, n_cat_feature))
    W1, W2 = W1/jnp.sqrt(n_num_feature + EPS), W2/jnp.sqrt(n_cat_feature + EPS)

    num_map, cat_map = batch.X[:, batch.num_idx] @ W1, batch.X[:, batch.cate_idx] @ W2
    rand_proj = jnp.concatenate([num_map, cat_map], axis=-1)
    normed_proj = rand_proj / (jnp.linalg.norm(rand_proj, axis=-1, keepdims=True) + EPS)

    # noisy sample
    Xt = jnp.sqrt(alpha_hats[t]) * batch.X + jnp.sqrt(1 - alpha_hats[t]) * eps_sample

    def diffusion_loss_fn(diffusion_paras: Params, embed_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        embed = embed_model.apply(embed_paras,
                                  batch.X + sigma * jax.random.normal(z_key, batch.X.shape),
                                  embed_t,  # timestep
                                  rngs={'dropout': drop_key1},
                                  training=True)

        pred_eps = noise_model.apply(diffusion_paras,
                                     Xt,
                                     t,
                                     embed,
                                     rngs={'dropout': drop_key2},
                                     training=True)

        code_vec, _, _ = codebook(embed)
        commitment_loss = (((code_vec - embed) ** 2).sum(axis=-1)).mean()

        normed_embed = embed / (jax.lax.stop_gradient(jnp.linalg.norm(embed, axis=-1, keepdims=True)) + EPS)

        # use normed L2 == cosine
        embed_dist = ((normed_embed - normed_embed[perm_idx]) ** 2).sum(axis=-1)
        rand_dist = ((normed_proj - normed_proj[perm_idx]) ** 2).sum(axis=-1)

        # random distance prediction loss
        rdm_loss = ((embed_dist - rand_dist) ** 2).mean()
        l1_norm = jnp.abs(embed).sum(axis=-1).mean()

        noise_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()
        if no_diffusion:
            loss = rdm_loss
        else:
            loss = noise_loss + alpha * rdm_loss

        return loss, {'loss': loss,
                      'noise_loss': noise_loss,
                      'rdm_loss': rdm_loss,
                      "l1_norm": l1_norm,
                      'commitment_loss': commitment_loss,
                      }

    def codebook_loss_fn(cb_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        embed = embed_model(batch.X, embed_t)
        code_vec, codes, l2_sum = codebook.apply(cb_params, embed)

        code_mtrx = cb_params['params']['codebook']
        ortho_loss = ((code_mtrx @ code_mtrx.transpose() - jnp.diag(jnp.ones(code_mtrx.shape[0]))) ** 2).mean()

        code_loss = (((code_vec - embed) ** 2).sum(axis=-1)).mean()

        loss = code_loss + ortho_loss

        return loss, {'codebook_loss': loss,
                      'code_loss': code_loss,
                      'ortho_loss': ortho_loss}

    new_noise_model, _ = noise_model.apply_gradient(lambda paras: diffusion_loss_fn(paras, embed_model.params))
    new_embed_model, info = embed_model.apply_gradient(lambda paras: diffusion_loss_fn(noise_model.params, paras))
    new_codebook, cb_info = codebook.apply_gradient(codebook_loss_fn)
    info.update(cb_info)

    return rng, new_noise_model, new_embed_model, new_codebook, info


@jax.jit
def jit_update_embedding(embed_model: Model,
                         metabatch: MetaBatch,
                         rng: PRNGKey,
                         T: int,
                         ) -> Tuple[PRNGKey, Model, InfoDict]:
    rng, dropout_key = jax.random.split(rng)
    batch_size = metabatch.X.shape[0]
    support_size = metabatch.X_support.shape[0]

    embed_t = jnp.ones((batch_size, 1)) * T

    def embed_loss_fn(embed_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        X_embed = embed_model.apply(embed_paras,
                                    metabatch.X,
                                    embed_t,
                                    rngs={'dropout': dropout_key},
                                    training=True)

        tar_embed = embed_model.apply(embed_paras,
                                      metabatch.X_tar,
                                      embed_t,
                                      rngs={'dropout': dropout_key},
                                      training=True)

        support_embed = embed_model.apply(embed_paras,
                                          metabatch.X_support,
                                          jnp.ones((support_size, 1)) * T,
                                          rngs={'dropout': dropout_key},
                                          training=True)

        partition = jnp.exp(-((X_embed[:, jnp.newaxis, :] - support_embed) ** 2).sum(axis=-1)).sum(axis=-1)
        cross_entropy_loss = - jnp.log(jnp.exp(-((X_embed - tar_embed) ** 2).sum(axis=-1)) / (partition + EPS)).mean()

        return cross_entropy_loss, {'cross_entropy_loss': cross_entropy_loss, }

    new_embed_model, info = embed_model.apply_gradient(embed_loss_fn)

    return rng, new_embed_model, info


class DiffusionFeatureExtractor(object):

    def __init__(self,
                 seed: int,
                 x_dim: int,
                 x_cat_counts: list,  # given by [num_cats, ...]
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 k_way: int,
                 # rand_proj_dim: int,
                 alpha: float,
                 # rand_proj_type: str,
                 std_scale: np.ndarray,
                 col_select_ratio: float = 0.5,
                 # lambda_range: tuple = (0.01, 1),
                 embed_t_dependent: bool = True,
                 lr: Union[float, optax.Schedule] = 3e-4,
                 no_diffusion: bool = False,
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 T: int = 10,  # number of backward steps
                 num_last_repeats: int = 0,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 100000,
                 sampler: str = "ddpm",
                 temperature: float = 1,
                 ):

        self.x_dim = x_dim
        self.x_cat_counts = x_cat_counts
        self.embed_dim = embed_dim
        self.k_way = k_way
        hidden_dims = (hidden_dim,) * num_layers

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)

        time_embedding = partial(FourierFeatures,
                                 output_size=time_dim,
                                 learnable=False)

        time_net = partial(MLP,
                           hidden_dims=(32, 32),
                           activations=mish,
                           activate_final=False)

        if lr_decay_steps is not None:
            lr = optax.cosine_decay_schedule(lr, lr_decay_steps)

        num_noise = partial(MLP,
                            hidden_dims=tuple(list(hidden_dims) + [x_dim]),
                            activations=mish,
                            layer_norm=layer_norm,
                            dropout_rate=dropout_rate,
                            activate_final=False)

        model_def = DDPM(time_embedding=time_embedding,
                         time_net=time_net,
                         noise_predictor=num_noise)

        embedding_model = partial(MLP,
                                  hidden_dims=tuple(list(hidden_dims) + [embed_dim]),
                                  activations=mish,
                                  layer_norm=layer_norm,
                                  dropout_rate=dropout_rate,
                                  activate_final=False)

        embed_def = DDPM(time_embedding=time_embedding,
                         time_net=time_net,
                         noise_predictor=embedding_model)

        embed_model = Model.create(embed_def,
                                   inputs=[key,
                                           jnp.zeros((1, x_dim)),  # x
                                           jnp.zeros((1, 1))],  # timestep
                                   optimizer=optax.adam(learning_rate=lr))

        tar_embed_model = Model.create(embed_def,
                                       inputs=[key,
                                               jnp.zeros((1, x_dim)),  # x
                                               jnp.zeros((1, 1))],  # timeste
                                       )
        noise_model = Model.create(model_def,  # state, action_t, timestep -> action_(t-1) || first dim=batch
                                   inputs=[key,
                                           jnp.zeros((1, x_dim)),  # x
                                           jnp.zeros((1, 1)),  # t
                                           jnp.zeros((1, embed_dim))],  # z(x)
                                   optimizer=optax.adam(learning_rate=lr))

        codebook = Model.create(CodeBook(embedding_dim=self.embed_dim, num_codes=self.embed_dim),
                                inputs=[key, jnp.zeros((1, self.embed_dim))],
                                optimizer=optax.adam(learning_rate=lr))

        # E(x0, t) or E(x0)
        self.noise_model = noise_model
        self.embed_model = embed_model
        self.codebook = codebook
        self.tar_embed_model = tar_embed_model
        self.embed_t_dependent = embed_t_dependent
        self.std_scale = std_scale
        self.col_select_ratio = col_select_ratio
        self.alpha = alpha
        self.no_diffusion = no_diffusion

        self.sampler = sampler
        self.temperature = temperature

        if beta_schedule == 'cosine':
            self.betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            self.betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            self.betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        self.betas = jnp.concatenate([jnp.zeros((1,)), self.betas])
        # add a special beginning beta[0] = 0 so that alpha[0] = 1, it is used for ddim

        alphas = 1 - self.betas
        self.alphas = alphas
        self.alpha_hats = jnp.cumprod(alphas)
        # sigma_min, sigma_max = lambda_range
        # self.lambda_ts = (sigma_min * (sigma_max / sigma_min) ** (jnp.arange(1, T + 1) / T)) ** 2
        # self.lambda_ts = jnp.array([0] * (T - 1) + [1])
        # self.lambda_ts / self.lambda_ts.sum()

        self.T = T
        self.num_last_repeats = num_last_repeats
        self.rng = rng

        self._n_training_steps = 0

    def update(self, batch) -> InfoDict:
        self.rng, self.noise_model, self.embed_model, self.codebook, info = (
            jit_update_diffusion_model_with_embedding(self.noise_model,
                                                      self.embed_model,
                                                      self.codebook,
                                                      self.embed_t_dependent,
                                                      self.alpha,
                                                      self.std_scale,
                                                      batch,
                                                      self.rng,
                                                      self.T,
                                                      self.alpha_hats,
                                                      self.col_select_ratio,
                                                      self.no_diffusion))

        self.tar_embed_model = self.tar_embed_model.replace(params=self.embed_model.params)
        self._n_training_steps += 1
        return info

    def meta_fine_tune(self, meta_batch: MetaBatch):
        self.rng, self.embed_model, info = jit_update_embedding(self.embed_model,
                                                                meta_batch,
                                                                self.rng,
                                                                self.T,
                                                                )
        self._n_training_steps += 1
        return info

    def generate(self, embedding: jnp.ndarray) -> jnp.ndarray:

        self.rng, key = jax.random.split(self.rng)
        prior = jax.random.normal(key, (embedding.shape[0], self.x_dim))
        x0, self.rng = ddpm_sampler(self.rng, self.noise_model.apply, self.noise_model.params, embedding, self.T,
                                    self.alphas, self.alpha_hats, sample_temperature=1,
                                    repeat_last_step=self.num_last_repeats, prior=prior)
        return x0

    def decoder(self, rng: PRNGKey,
                noise_model_fn: Callable,
                noise_model_params: Params,
                embedding: jnp.ndarray):
        rng, key = jax.random.split(rng)
        prior = jax.random.normal(key, (embedding.shape[0], self.x_dim))
        x0, rng = ddpm_sampler(rng, noise_model_fn, noise_model_params, embedding, self.T,
                               self.alphas, self.alpha_hats, sample_temperature=1,
                               repeat_last_step=self.num_last_repeats, prior=prior)
        return rng, x0

    def get_embedding(self, X: jnp.ndarray, time_level: int = None, use_tar=False, ):

        if self.embed_t_dependent:
            assert time_level is not None
            time = jnp.ones((X.shape[0], 1)) * time_level
        else:
            time = jnp.zeros((X.shape[0], 1))
        if use_tar:
            return self.tar_embed_model(X, time)
        return self.embed_model(X, time)  # [:, :self.embed_dim]  # [:, :self.embed_dim]
