import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import OpenmlDataset
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_embed_tsne(x_embeddings, y):
    tsne_embed = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=30).fit_transform(x_embeddings)

    df_ = pd.DataFrame(np.concatenate([tsne_embed, y[:, np.newaxis]], axis=-1),
                       columns=['x1', 'x2', 'y'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="x1", y="x2",
        hue="y",
        palette=sns.color_palette("hls", data.num_classes),
        data=df_,
        legend="full",
        alpha=1
    )
    plt.show()


if __name__ == '__main__':

    d_name = 'dna'
    data = OpenmlDataset(d_name, one_hot=True)

    # batch = data.sample_few_shot_support(50)
    batch = data.sample(min(data.train_size, 10000))

    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=50).fit_transform(batch.X)

    # X_embedded = PCA(n_components=2).fit_transform(batch.X)

    df = pd.DataFrame(np.concatenate([X_embedded, batch.y[:, np.newaxis]], axis=-1),
                      columns=['z1', 'z2', 'y'])

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="z1", y="z2",
        hue="y",
        palette=sns.color_palette("hls", data.num_classes),
        data=df,
        legend=None,
        alpha=1
    )
    plt.savefig(f'raw_{d_name}.png')
    plt.show()

