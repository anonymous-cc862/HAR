import jax
import jax.numpy as jnp
from functools import partial
from networks.types import Callable, Batch


@partial(jax.jit, static_argnames=('adaptive_steps', 'temperature'))
def jit_soft_kmeans(query_embedding: jnp.ndarray,  # (B, embed_dim)
                    c0: jnp.ndarray,  # (n_way x num_shot, embed_dim)
                    adaptive_steps: int,
                    temperature: float
                    ):
    """
    it returns instance-specific labels: indicating the [label]-th item of c0
    """
    def adaptive_proto_fn(ct, _):  # (n_way, embed_dim)
        logit = jax.vmap(lambda z: jnp.exp(-((z - ct) ** 2/temperature).sum(axis=-1)))(query_embedding)  # (B, embed_dim) -> (B, n_way)
        confid = logit / logit.sum(axis=-1, keepdims=True)  # (B, n_way) confidence weight matrix, normed along n_way
        sum_qf = jax.vmap(lambda q: (q[:, jnp.newaxis] * query_embedding).sum(axis=0))(confid.transpose())  # (n_way, embed_dim)
        c_t_new = (ct + sum_qf) / (1 + confid.transpose().sum(axis=-1, keepdims=True))
        return c_t_new, ()  # (n_way, embed_dim)
    proto_embeddings, () = jax.lax.scan(adaptive_proto_fn,
                                        c0,
                                        jnp.arange(adaptive_steps, 0, -1),
                                        unroll=5)
    scores = jax.vmap(lambda z: jnp.exp(-((z - proto_embeddings) ** 2/temperature).sum(axis=-1)))(query_embedding)
    labels = scores.argmax(axis=-1)

    return scores, labels


def iterative_protonet_predict(embed_fn: Callable,
                               support_set: Batch,
                               n_way: int,
                               num_shot: int,
                               query_x: jnp.array,
                               # unlabeled_x: jnp.array,
                               iteration_steps,
                               generator: Callable = None,
                               num_samples: int = 10,
                               temperature: float = 1,
                               mean_center: bool = False) -> tuple[jnp.array, jnp.array]:

    assert all(jnp.sort(support_set.y) == support_set.y), "The support set should be in order of [0,1,2,...]"
    query_num = query_x.shape[0]
    query_embed = embed_fn(query_x)  # (B, z_dim)
    # query_embed = embed_fn(jnp.concatenate([query_x, unlabeled_x], axis=0))  # (B, z_dim)
    s_embed = embed_fn(support_set.X)  # (num_shot x n_way, z_dim) [0,0,0, 1,1,1, 2,2,2, ...] (n_way, num_shot,-1)

    c0 = s_embed
    # if generator is None:
    #     c0 = s_embed  # .reshape(n_way, num_shot, -1).mean(axis=1)  # requires proper order of support samples
    if generator is not None:
        rep_s_embed = jnp.repeat(s_embed, num_samples, axis=0)
        x0 = generator(rep_s_embed)  # (n_way x num_shot x num_samples)
        x0_embed = embed_fn(x0)
        query_embed = jnp.concatenate([query_embed, x0_embed], axis=0)  # expand query embeddings
        # x0_embed = embed_fn(x0).reshape(n_way, num_shot * num_samples, -1)
        # c0 = jnp.concatenate([x0_embed, s_embed.reshape(n_way, num_shot, -1)], axis=1)  # .mean(axis=1)

    if mean_center:
        c0 = c0.reshape(n_way, num_shot, -1).mean(axis=1)
        y = jnp.arange(n_way)
    else:
        y = support_set.y.repeat((num_samples + 1))  # (num_samples + 1) * num_shot

    scores, labels = jit_soft_kmeans(query_embed,
                                     c0,
                                     iteration_steps,
                                     temperature,
                                     )
    # # #######test
    # # 添加嵌入计算步骤（关键修复）
    # support_emb = embed_fn(support_set.X)
    
    # # 初始化原型
    # prototypes = jnp.stack([jnp.mean(support_emb[support_set.y == k], axis=0) for k in range(n_way)])
    # ########
    return scores, y[labels][:query_num]

    # query_embed = embed_fn(query_x)  # (B, z_dim)
    # s_embed = embed_fn(support_set.X)  # (n_way, num_shot, z_dim) [0,0,0, 1,1,1, 2,2,2, ...] (n_way, num_shot,-1)
    # # c0 = jnp.array([s_embed[support_set.y == c_].mean(axis=0) for c_ in range(n_way)])
    # # expand support set
    # # num_shot = sum(support_set.y == 0)
    # # centers = s_embed.reshape(n_way, -1, s_embed.shape[-1]).mean(axis=1)  # (n_way, z_dim)
    # # dist = jax.vmap(lambda z: ((z - centers) ** 2).sum(axis=-1))(query_embed)  # (B, n_way)
    # #
    # # expanded_embed = jnp.concatenate([s_embed[jnp.argsort(dist[:, _])[:k]][jnp.newaxis, :] for _ in range(n_way)])
    # # s_embed = jnp.concatenate([s_embed.reshape(n_way, -1, s_embed.shape[-1]),
    # #                            expanded_embed.reshape(n_way, -1, s_embed.shape[-1])], axis=1).reshape(-1, s_embed.shape[-1])
    # # y = jnp.arange(n_way).repeat(num_shot+k)
    #
    #
    # if generator is None:
    #     c0 = s_embed  # .reshape(n_way, num_shot, -1).mean(axis=1)  # requires proper order of support samples
    #     # y = support_set.y
    # else:
    #     # num_shot = sum(support_set.y == 0)
    #     rep_s_embed = jnp.repeat(s_embed, num_samples, axis=0)
    #     # y = support_set.y.repeat((num_samples + 1))  # num_samples + support
    #     y = y.repeat((num_samples + 1))  # num_samples + support
    #     x0 = generator(rep_s_embed)  # (n_way x num_shot x num_samples)
    #     # rng, x0 = decoder(rng, noise_model_fn, noise_model_params, rep_s_embed)
    #     x0_embed = embed_fn(x0).reshape(n_way, -1, s_embed.shape[-1])
    #     # concatenate samples embeddings and support embeddings #  TODO: don't take average: use multi-modality adaptive protonet
    #     c0 = jnp.concatenate([x0_embed, s_embed.reshape(n_way, -1, s_embed.shape[-1])], axis=1)  # .mean(axis=1)
    #     c0 = c0.reshape(-1, s_embed.shape[-1])
    #     # num_shot += num_samples  # the samples for center also count the generated samples
    #
    # scores, labels = jit_adaptive_predict(query_embed,
    #                                       c0,
    #                                       # num_shot,
    #                                       adaptive_steps,
    #                                       temperature,
    #                                       )
    # return scores, y[labels]


def diffusion_protonet_predict(embed_fn: Callable,
                               support_set: Batch,
                               n_way: int,
                               query_x: jnp.array,
                               generator: Callable,
                               num_samples: int = 10) -> tuple[jnp.array, jnp.array]:
    # TODO: generate more support samples
    num_shot = sum(support_set.y == 0)
    query_embed = embed_fn(query_x)  # (B, z_dim)
    s_embed = embed_fn(support_set.X)  # (num_shot x n_way, z_dim) [0,0,0, 1,1,1, 2,2,2, ...] (n_way, num_shot,-1)
    # (n_way x num_shot x rep,-1) can be reshaped to (n_way, num_shot x rep,-1)
    rep_s_embed = jnp.repeat(s_embed, num_samples, axis=0)
    x0 = generator(rep_s_embed)
    # rng, x0 = decoder(rng, noise_model_fn, noise_model_params, rep_s_embed)
    x0_embed = embed_fn(x0).reshape(n_way, num_shot * num_samples, -1)
    # concatenate samples embeddings and support embeddings
    c0 = jnp.concatenate([x0_embed, s_embed.reshape(n_way, num_shot, -1)], axis=1).mean(axis=1)
    scores = jax.vmap(lambda z: jnp.exp(-((z - c0) ** 2).sum(axis=-1)))(query_embed)
    labels = scores.argmax(axis=-1)

    return scores, labels


# class ProtoNet(object):
#     def __init__(self, adaptive_steps: int = 0):
#         # self.embed_fn = embed_fn
#         self.adaptive_steps = adaptive_steps
#         self.proto_embeddings = None
#
#     # def build_clf(self, support_set: Batch, n_way: int):
#     #     # protos = []
#     #
#     #     embeddings = self.embed_fn(support_set.X)
#     #     self.proto_embeddings = jnp.array([embeddings[support_set.y == c_].mean(axis=0) for c_ in range(n_way)])
#
#     def predict(self, embed_fn: Callable,
#                 support_set: Batch,
#                 n_way: int,
#                 X: jnp.array,
#                 adaptive_steps) -> tuple[jnp.array, jnp.array]:
#         # supp_embeddings = self.embed_fn(support_set.X)
#         X_embed = embed_fn(X)  # (B, embed_dim)
#         S_embed = embed_fn(support_set.X)
#         num_shot = sum(support_set.y == 0)
#
#         # (n_way, embed_dim)
#         c0 = jnp.array([S_embed[support_set.y == c_].mean(axis=0) for c_ in range(n_way)])
#
#         score, labels = jit_predict(X_embed,
#                                     c0,
#                                     num_shot,
#                                     adaptive_steps
#                                     )
#         return score, labels

# def adaptive_proto_fn(ct, _):
#     logit = jax.vmap(lambda z: jnp.exp(-((z - ct) ** 2).sum(axis=-1)))(X_embed)  # (B, n_way)
#     qs = logit / logit.sum(axis=-1, keepdims=True)  # (B, n_way)
#     sum_qf = jax.vmap(lambda q: (q[:, jnp.newaxis] * X_embed).sum(axis=0))(qs.transpose())
#     c_t_new = (c0 * num_shot + sum_qf) / (num_shot + qs.transpose().sum(axis=-1, keepdims=True))
#     return c_t_new, ()
#
# proto_embeddings, () = jax.lax.scan(adaptive_proto_fn,
#                                     c0,
#                                     jnp.arange(self.adaptive_steps, 0, -1),
#                                     unroll=5)
#
# scores = jax.vmap(lambda z: jnp.exp(-((z - proto_embeddings) ** 2).sum(axis=-1)))(X_embed)
# labels = scores.argmax(axis=-1)
# return scores, labels
