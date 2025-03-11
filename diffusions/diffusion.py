from functools import partial
from typing import Type, Tuple
from networks.model import Model
from networks.types import Params, InfoDict, PRNGKey, Batch
import flax.linen as nn
import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def rescale_noise_cfg(cfg_noise: jnp.array, cond_noise: jnp.array, guidance_rescale: float = 0.7):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    # skip the batch dimensions, and calculate the noise of all place
    std_cond = cond_noise.std(axis=list(range(1, cond_noise.ndim)), keepdims=True)
    std_cfg = cfg_noise.std(axis=list(range(1, cfg_noise.ndim)), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_rescaled = cfg_noise * (std_cond / std_cfg + 1e-6)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    cfg_noise = guidance_rescale * noise_rescaled + (1 - guidance_rescale) * cfg_noise
    return cfg_noise


class DDPM(nn.Module):
    noise_predictor: Type[nn.Module]
    time_embedding: Type[nn.Module]
    time_net: Type[nn.Module]
    """
    if cond_embedding:
        treat the last dim as the conditional tag
    """

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,  # s = [obs, cond] if it's conditional
                 time: jnp.ndarray,
                 prompt: jnp.ndarray = None,
                 training: bool = False):
        t_ff = self.time_embedding()(time)
        time_suffix = self.time_net()(t_ff, training=training)

        if prompt is None:
            reverse_input = jnp.concatenate([x, time_suffix], axis=-1)
        else:
            reverse_input = jnp.concatenate([x, time_suffix, prompt], axis=-1)
        return self.noise_predictor()(reverse_input, training=training)


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step'))
def ddpm_sampler(rng, noise_pred_apply_fn, params, prompts, T, alphas, alpha_hats, sample_temperature,
                 repeat_last_step, prior: jnp.array):
    batch_size = prompts.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        input_time = input_time_proto * t
        # noise_model(s, a, time, training=training) in DDPM

        eps_pred = noise_pred_apply_fn(params, current_x, input_time, prompts, training=False)

        alpha_1 = 1 / jnp.sqrt(alphas[t])
        alpha_2 = ((1 - alphas[t]) / (jnp.sqrt(1 - alpha_hats[t])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_, key_ = jax.random.split(rng_, 2)

        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # sigmas_t = jnp.sqrt((1 - alphas[t]) * (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]))
        sigmas_t = jnp.sqrt((1 - alphas[t]))  # both have similar results
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (sigmas_t * z_scaled)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=5)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    return output_tuple


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'ddim_step',
                                   'ddim_eta'))
def ddim_sampler(noise_pred_apply_fn, params, T, rng, obs_with_prompt, alphas, alpha_hats,
                 sample_temperature, repeat_last_step, prior: jnp.array,
                 ddim_step=1, ddim_eta=0):
    """
    dim(obs_with_prompt) = dim(obs) + 1, the prompt is one scalar value
    """
    batch_size = obs_with_prompt.shape[0]
    c = T // ddim_step  # jump step
    ddim_time_seq = jnp.concatenate([jnp.arange(T, 0, -c), jnp.array([0])])

    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, i):
        # work on the last dim
        current_x, rng_ = input_tuple

        t, prev_t = ddim_time_seq[i], ddim_time_seq[i + 1]

        input_time = input_time_proto * t

        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        eps_pred = noise_pred_apply_fn(params, obs_with_prompt, current_x, input_time, training=False)

        # sigmas_t = ddim_eta * jnp.sqrt((1 - alpha_hats[prev_t]) / (1 - alpha_hats[t]) * (1 - alphas[t]))
        sigmas_t = ddim_eta * jnp.sqrt((1 - alphas[t]))  # both have similar results

        alpha_1 = 1 / jnp.sqrt(alphas[t])
        alpha_2 = jnp.sqrt(1 - alpha_hats[t])
        alpha_3 = jnp.sqrt(1 - alpha_hats[prev_t] - sigmas_t ** 2)

        current_x = alpha_1 * (current_x - alpha_2 * eps_pred) + alpha_3 * eps_pred

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z
        current_x = current_x + sigmas_t * z_scaled

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn, (prior, denoise_key), jnp.arange(len(ddim_time_seq) - 1),
                                    unroll=5)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    x0, rng = output_tuple

    return x0, rng


@jax.jit
def jit_update_diffusion_model(noise_model: Model,
                               batch: Batch,
                               rng: PRNGKey,
                               T,
                               alpha_hats) -> Tuple[PRNGKey, Tuple[Model, InfoDict]]:
    rng, t_key, noise_key, tr_key = jax.random.split(rng, 4)
    # learnable t is ranged from 1, 2,...,T corresponding to the indices of alphas
    # assert len(alpha_hat) == T + 1
    t = jax.random.randint(t_key, (batch.x.shape[0],), 1, T + 1)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch.x.shape)

    # noisy_samples = jnp.sqrt(alpha_hat[t]) * batch.actions + (1 - jnp.sqrt((alpha_hat[t]))) * eps_sample
    noisy_samples = jnp.sqrt(alpha_hats[t]) * batch.x + jnp.sqrt(1 - alpha_hats[t]) * eps_sample

    def actor_loss_fn(paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = noise_model.apply(paras,
                                     noisy_samples,
                                     t,  # noice that t is ranged from 0, 1, ..., T-1, pay attention to sampling method
                                     rngs={'dropout': tr_key},
                                     training=True)

        noise_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()

        return noise_loss, {'noise_loss': noise_loss}

    return rng, noise_model.apply_gradient(actor_loss_fn)
