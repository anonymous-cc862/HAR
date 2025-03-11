import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import datetime
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import trange
from typing import Callable
from protonet import iterative_protonet_predict
from datasets import OpenmlDataset
from diffusion_feature import DiffusionFeatureExtractor
from tensorboardX import SummaryWriter
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 520, 'Seed.')
flags.DEFINE_integer('max_epochs', 70000, 'Max number of training epochs')#] #150000
flags.DEFINE_integer('min_accepted_step', 50000, 'Minimum number of training epochs considered as accepted')
flags.DEFINE_string('data', 'dna', "choose data")
flags.DEFINE_enum('norm', 'minmax', ['std', 'minmax', 'no'],
                  'Choose how to normalize the numerical features')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('T', 10, 'Total diffusion steps')
flags.DEFINE_integer('embed_dim', 10, 'The embedding dimension')
flags.DEFINE_integer('hidden_dim', 3000, 'The hidden dimension of neural networks')
flags.DEFINE_integer('num_layers', 1, 'The num of layers of neural networks')
# flags.DEFINE_integer('rand_proj_dim', 10, 'The dimension of random projection')
flags.DEFINE_float('dropout', 0., 'The dropout rate')
flags.DEFINE_integer('embed_lvl', 10,
                     'The time step used as embedding when t_dependent_embedding=True')
flags.DEFINE_float('alpha', 0.5, 'The weight of random distance prediction loss')
flags.DEFINE_float('noise_aug_scale', 1., 'X0 += noise_aug_scale * std * N(0, 1), =0 means no aug,')
flags.DEFINE_float('col_select_ratio', 0.1, 'select col with prob = col_select_ratio.')
flags.DEFINE_bool('t_dependent_embedding', False, 'Whether the embedding is dependent on t or not')
flags.DEFINE_bool('one_hot', True, 'Whether use one hot encoding for categorical variables')
flags.DEFINE_integer('test_interval', 10000, 'Interval of epochs between tests')
flags.DEFINE_integer('plot_interval', 50000, 'Interval of epochs between tests')
flags.DEFINE_integer('num_test_runs', 100, 'Number of test runs for evaluation')
flags.DEFINE_integer('log_interval', 100, 'Interval of epochs between logs')
flags.DEFINE_integer('num_shot', 3, 'Number of few shot for support set in each classes')
flags.DEFINE_integer('iteration_steps', 5, 'The number of iteration_steps steps for prototype iteration')
flags.DEFINE_float('temperature', 0.5, 'The weight of random distance prediction loss')
flags.DEFINE_bool('mean_center', False, 'Whether use mean center of each class in ProtoNet')
flags.DEFINE_bool('no_diffusion', False, 'Whether to disable diffusion loss')
flags.DEFINE_bool('tsne', False, 'Whether to create tsne plots during the training')
flags.DEFINE_integer('num_samples', 0, 'The number of samples for generated for ProtoNet')
flags.DEFINE_string('tag', '', 'Give some tags')
flags.DEFINE_integer('gpu', 3, 'Which gpu to use for')


def main(_):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    # set seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # fetch dataset
    data = OpenmlDataset(FLAGS.data, one_hot=FLAGS.one_hot, norm=FLAGS.norm)
    query_set = data.get_test_batch()
    data.summary()
    # suffix = f'{FLAGS.num_shot}Shot_a={FLAGS.alpha}_zdim={FLAGS.embed_dim}_{FLAGS.tag}'
    suffix = f'{FLAGS.num_shot}-Shot-{FLAGS.tag}'

    time_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_dir = os.path.join('results', 'tensorboard', f'{data.data_name}', f'{time_str}-{suffix}')
    summary_writer = SummaryWriter(save_dir)

    # record the config file
    print('=' * 10 + ' Arguments ' + '=' * 10)
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        for k, v in FLAGS.flag_values_dict().items():
            value = str(v)
            print(k + ' = ' + value)
            print(k + ' = ' + value, file=file)
        print(f"Save_folder = {save_dir}", file=file)

    with open(os.path.join(save_dir, "accuracies.txt"), "w") as f:
        print("\t".join(["steps", "mean", "std", "min", "max", "val", "final"]), file=f)

    print(f'Save results to:{save_dir}')

    def plot_embed_tsne(x_embeddings, y, save_path=''):
        tsne_embed = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=50).fit_transform(x_embeddings)

        df_ = pd.DataFrame(np.concatenate([tsne_embed, y[:, np.newaxis]], axis=-1),
                           columns=['z1', 'z2', 'y'])
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x="z1", y="z2",
            hue="y",
            palette=sns.color_palette("hls", data.num_classes),
            data=df_,
            legend=None,
            alpha=1
        )
        plt.savefig(save_path)
        plt.show()

    def test_protonet(embed_fn: Callable,
                      data_: OpenmlDataset,
                      num_shot: int,
                      num_runs: int,
                      generator: Callable = None,
                      num_samples: int = 10):
        ts_acc, val_acc = [], []
        for _ in range(num_runs):
            # create classifier using protonet
            support_set = data_.sample_few_shot_support(num_shot)
            # unlabeled_set = data_.sample(256)
            scores, labels = iterative_protonet_predict(embed_fn,
                                                        support_set=support_set,
                                                        n_way=data_.num_classes,
                                                        num_shot=FLAGS.num_shot,
                                                        query_x=query_set.X,
                                                        iteration_steps=FLAGS.iteration_steps,
                                                        generator=generator,
                                                        num_samples=num_samples,
                                                        temperature=FLAGS.temperature,
                                                        mean_center=FLAGS.mean_center)
            val_s, val_q = data_.sample_val_batch()
            val_scores, val_labels = iterative_protonet_predict(embed_fn,
                                                                support_set=val_s,
                                                                n_way=data_.num_classes,
                                                                num_shot=1,  # always using one-hot to validation
                                                                query_x=val_q.X,
                                                                iteration_steps=FLAGS.iteration_steps,
                                                                generator=generator,
                                                                num_samples=num_samples,
                                                                temperature=FLAGS.temperature,
                                                                mean_center=FLAGS.mean_center)

            ts_acc.append((labels == query_set.y).sum() / len(labels))
            val_acc.append((val_labels == val_q.y).sum() / len(val_labels))
        return ts_acc, val_acc

    model = DiffusionFeatureExtractor(FLAGS.seed,
                                      x_dim=len(data.all_features),
                                      x_cat_counts=[],  # len(data.cat_features),
                                      embed_dim=FLAGS.embed_dim,
                                      hidden_dim=FLAGS.hidden_dim,
                                      num_layers=FLAGS.num_layers,
                                      k_way=data.num_classes,
                                      # rand_proj_dim=FLAGS.rand_proj_dim,
                                      alpha=FLAGS.alpha,
                                      # rand_proj_type=FLAGS.proj,
                                      dropout_rate=FLAGS.dropout,
                                      std_scale=data.X_stds * FLAGS.noise_aug_scale,
                                      embed_t_dependent=FLAGS.t_dependent_embedding,
                                      no_diffusion=FLAGS.no_diffusion,
                                      T=FLAGS.T,
                                      col_select_ratio=FLAGS.col_select_ratio,
                                      )

    # train
    best_val, final_test_acc = 0, 0  # validation is used for early stopping
    for i in trange(FLAGS.max_epochs + 1, desc='Training'):

        if i % FLAGS.test_interval == 0:
            # record reconstruction error
            embed = model.get_embedding(query_set.X, time_level=FLAGS.embed_lvl)
            x0_hat = model.generate(embed)
            test_error = ((query_set.X - x0_hat) ** 2).sum(axis=-1).mean()

            # test accuracy with ProtoNet
            acc, val_acc = test_protonet(lambda x: model.get_embedding(x, time_level=FLAGS.embed_lvl),
                                         data_=data,
                                         num_shot=FLAGS.num_shot,
                                         num_runs=FLAGS.num_test_runs,
                                         generator=model.generate if FLAGS.num_samples > 0 else None,
                                         num_samples=FLAGS.num_samples)

            acc_mean, acc_std, acc_min, acc_max = (np.mean(acc), np.std(acc),
                                                   np.min(acc), np.max(acc))

            mean_val_acc, std_val_acc = np.mean(val_acc), np.std(val_acc)
            if mean_val_acc > best_val and i >= FLAGS.min_accepted_step:
                best_val = mean_val_acc
                final_test_acc = acc_mean
            print(f"Step[{i}]: Test_acc={round(acc_mean, 5)} (std={round(acc_std, 5)}) "
                  f"| Val_acc={round(mean_val_acc, 5)} (std={round(std_val_acc, 5)})"
                  f"| Final_acc={round(final_test_acc, 5)}*")
            # save results in the file
            with open(os.path.join(save_dir, "accuracies.txt"), "a+") as f:
                print("\t".join([f"{i}",
                                 f"{round(acc_mean, 5)}",
                                 f"{round(acc_std, 5)}",
                                 f"{round(acc_min, 5)}",
                                 f"{round(acc_max, 5)}",
                                 f"{round(mean_val_acc, 5)}",
                                 f"{round(final_test_acc, 5)}*"]), file=f)

            summary_writer.add_scalar(f'test/accuracy', acc_mean, i)
            summary_writer.add_scalar(f'test/accuracy_std', acc_std, i)
            summary_writer.add_scalar(f'test/accuracy_min', acc_min, i)
            summary_writer.add_scalar(f'test/accuracy_max', acc_max, i)
            summary_writer.add_scalar(f'test/recon_error', test_error, i)
            summary_writer.flush()

        if FLAGS.tsne and i % FLAGS.plot_interval == 0:
            # create t-sne plots of the embedding
            plot_batch = data.sample(min(data.train_size, 1000))
            plot_x_embed = model.get_embedding(plot_batch.X, time_level=FLAGS.embed_lvl)
            plot_embed_tsne(plot_x_embed, plot_batch.y, save_path=os.path.join(save_dir, f'tsne{i}.png'))

        batch = data.sample(FLAGS.batch_size)
        info = model.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
