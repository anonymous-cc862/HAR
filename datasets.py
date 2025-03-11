import warnings

import openml
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from networks.types import Batch, Union
from protonet import jit_soft_kmeans

std_norm, rob_norm = StandardScaler(), RobustScaler()

DATASET_NAMES = {
    "cmc": {'target': "Contraceptive_method_used"},
    "mfeat-karhunen": {'target': "class"},
    "optdigits": {'target': "class"},
    "diabetes": {'target': "class"},
    # "semeion": {'target': "Class", "type": "categorical"},  # more like categorical
    "semeion": {'target': "Class"},  # more like categorical
    "mfeat-pixel": {'target': "class", "type": "numerical"},  # numerical!
    "dna": {'target': "class"},
    "income": {'target': "class"},
    "Car": {'target': "binaryClass"},
    "Bank_marketing_data_set_UCI": {'target': "y"},
    "jungle_chess_2pcs_raw_endgame_complete": {'target': "class", },
    "mushroom":  {'target': "class"},
    "soybean": {'target': "class"},
    "abalone": {'target': "Class_number_of_rings"},
    "california":  {'target': "price"},
    "mnist_784": {'target': "class"},
    "letter": {'target': "class"},
    "nomao": {'target': "Class"},
    "heart": {'target': "HeartDisease"},
    "human_activity": {'target': "class"}
}


def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""
    """it is faster to get replace=False"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


# def in_batch_sample(batch: Batch, num_samples: int, y_label=None) -> Batch:
#     indices = sample_n_k(len(batch.X), num_samples)
#     if y_label == 'pseudo':
#         y = np.arange(len(indices))
#     else:
#         y = batch.y[indices]
#     return Batch(X=batch.X[indices],  # concat(num, cat)
#                  cate_mask=self.cate_mask.repeat(len(indices), axis=0),
#                  y=y)

def load_heart_dataset():
    pd.read_csv("data/heart.csv")


def load_income_dataset():
    x_tr = np.load("data/income/xtrain.npy")
    y_tr = np.load("data/income/ytrain.npy")
    x_ts = np.load("data/income/xtest.npy")
    y_ts = np.load("data/income/ytest.npy")

    X, y = np.concatenate([x_tr, x_ts]), np.concatenate([y_tr, y_ts])
    Xy = pd.DataFrame(np.concatenate([x_tr, x_ts]), columns=[f'V{_}' for _ in range(X.shape[1])])
    Xy['class'] = y
    for col in Xy:
        if len(Xy[col].unique()) <= 2:
            Xy[col] = Xy[col].astype('category')

    tr_idx, ts_idx = np.arange(len(x_tr)), np.arange(len(x_ts), len(X))

    return Xy, tr_idx, ts_idx

def load_human_activity_dataset():
    x_tr = np.loadtxt('data/human_activity/X_train.txt') #np.load("data/income/xtrain.npy")
    y_tr = np.loadtxt('data/human_activity/y_train.txt')
    x_ts = np.loadtxt('data/human_activity/X_test.txt')
    y_ts = np.loadtxt('data/human_activity/y_test.txt')

    X, y = np.concatenate([x_tr, x_ts]), np.concatenate([y_tr, y_ts])
    Xy = pd.DataFrame(np.concatenate([x_tr, x_ts]), columns=[f'V{_}' for _ in range(X.shape[1])])
    Xy['class'] = y
    for col in Xy:
        if len(Xy[col].unique()) <= 2:
            Xy[col] = Xy[col].astype('category')

    tr_idx, ts_idx = np.arange(len(x_tr)), np.arange(len(x_tr), len(X))

    return Xy, tr_idx, ts_idx



class OpenmlDataset(object):
    def __init__(self, name: str,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 one_hot: bool = False,
                 cate_perturbation_scale: float = 0.1,
                 norm: str = 'std',  # 'std', 'minmax', 'no'
                 ):

        # fuzzy dataset search
        match_names = [n_ for n_ in DATASET_NAMES if n_.find(name) > -1]
        if not match_names:
            warnings.WarningMessage(f"Only support dataset in {DATASET_NAMES} (unrecognized dataset: {name})")
        # assert match_names, f"Only support dataset in {DATASET_NAMES} (unrecognized dataset: {name})"
        data_name = match_names[0]
        print(f"Use dataset={data_name}")
        if data_name == 'income':
            Xy, self.train_indices, self.test_indices = load_income_dataset()
            num_val = int(val_size * len(self.train_indices))
            self.train_indices, self.val_indices = self.train_indices[:-num_val], self.train_indices[-num_val:]
        elif data_name == 'heart':
            Xy = pd.read_csv("data/heart.csv")
            self.train_indices, self.test_indices, self.val_indices = None, None, None
        elif data_name == 'human_activity':
            Xy, self.train_indices, self.test_indices = load_human_activity_dataset()
            num_val = int(val_size * len(self.train_indices))
            self.train_indices, self.val_indices = self.train_indices[:-num_val], self.train_indices[-num_val:]
        else:
            dataset = openml.datasets.get_dataset(data_name)
            Xy, _, _, _ = dataset.get_data(dataset_format="dataframe")
            self.train_indices, self.test_indices, self.val_indices = None, None, None

        self.data_name = data_name
        self.target = DATASET_NAMES[data_name]['target']
        if 'type' in DATASET_NAMES[data_name]:
            data_type = DATASET_NAMES[data_name]['type']
            if data_type == 'numerical':
                Xy = Xy.astype(float)
                Xy[self.target] = Xy[self.target].astype('category')
            elif data_type == 'categorical':
                Xy = Xy.astype('category')

        self.cat_features = []
        self.cat_num = []
        self.num_features = []
        for col in Xy.columns:
            if Xy[col].dtype.kind == 'O' or col == self.target:
                Xy[col] = Xy[col].astype('category')
                if col != self.target:
                    self.cat_features.append(col)
            else:
                self.num_features.append(col)

            # if col != self.target:
            #     if Xy[col].dtype.kind == 'O':
            #         Xy[col] = Xy[col].astype('category')
            #     # if hasattr(Xy[col], 'cat'):
            #         self.cat_features.append(col)
            #     else:
            #         self.num_features.append(col)

        self.X_num, self.X_cat, self.y = (Xy[self.num_features], Xy[self.cat_features], Xy[self.target].cat.codes)

        # shuffle idx
        # cols = self.X_num.columns.tolist()
        # random.shuffle(cols)
        # self.X_num = self.X_num[cols]

        # one_hot embedding of categorical features: only encode binary features
        if self.cat_features and one_hot:
            not_binary = [c_ for c_ in self.cat_features if len(self.X_cat[c_].cat.categories) > 2]
            # self.X_cat = self.X_cat.apply(lambda x: x.cat.codes)
            if not_binary:
                one_hot_cat = pd.get_dummies(self.X_cat[not_binary]).astype(int)
                binary_cat = self.X_cat.drop(not_binary, axis=1).apply(lambda x: x.cat.codes)
                self.X_cat = pd.concat([binary_cat, one_hot_cat], axis=1)
            else:
                self.X_cat = self.X_cat.apply(lambda x: x.cat.codes)

            self.cat_features = self.X_cat.columns.tolist()
            self.cat_num = [2] * len(self.cat_features)

        elif self.cat_features:
            self.cat_num = [len(self.X_cat[c_].cat.categories) for c_ in self.cat_features]
            self.X_cat = self.X_cat.apply(lambda x: x.cat.codes)

        self.all_features = self.num_features + self.cat_features
        self.num_idx = np.arange(len(self.num_features))
        self.cat_idx = np.arange(len(self.num_features), len(self.all_features))
        # self.cate_mask = np.array([0] * len(self.num_features) + [1] * len(self.cat_features))[np.newaxis, :]

        self.size = len(Xy)
        self.labels = np.unique(self.y)
        self.num_classes = len(self.labels)  # len(set(self.y))
        if self.train_indices is None or self.test_indices is None:
            self.train_indices, self.test_indices = train_test_split(np.arange(self.size), test_size=test_size)
            # to make sure the training set contains samples from each classes
            shuffle_count = 0
            while len(set(self.y[self.test_indices])) < self.num_classes and shuffle_count < 10000:
                self.train_indices, self.test_indices = train_test_split(np.arange(self.size), test_size=test_size)
                shuffle_count += 1
            num_val = int(val_size * len(self.train_indices))
            self.train_indices, self.val_indices = self.train_indices[:-num_val], self.train_indices[-num_val:]
            assert shuffle_count < 10000, "Cannot find a train-test split with training set contains all classes"

        self.train_size, self.test_size = len(self.train_indices), len(self.test_indices)

        if norm == 'std' or data_name == 'income':
            # normalize numerical features
            self.X_num = (self.X_num - self.X_num.mean()) / (1e-6 + self.X_num.std())
        elif norm == 'minmax':
            self.X_num = (self.X_num - self.X_num.min()) / (1e-6 + self.X_num.max() - self.X_num.min())

        self.X_stds = np.concatenate([self.X_num.std().to_numpy(),
                                      np.ones(len(self.cat_features)) * cate_perturbation_scale])

    def sample(self, batch_size: int) -> Batch:
        indices = self.train_indices[sample_n_k(self.train_size, batch_size)]
        return self.get_batch_data(indices)

    def sample_few_shot_support(self, n_shot: int) -> Batch:
        # the few shot samples are samples from training set with disclosed labels
        # y is ordered as [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...] (e.g. 5-shot)
        sampled_indicis = []
        for i in range(self.num_classes):
            i_cls_indices = self.y[self.train_indices].index.to_numpy()[self.y[self.train_indices] == i]
            assert len(i_cls_indices) >= n_shot, f"The number of samples from class={i} < few_show={n_shot}"
            sampled_indicis += np.random.choice(i_cls_indices, n_shot, replace=False).tolist()
        return self.get_batch_data(sampled_indicis)

    def sample_val_batch(self) -> tuple[Batch, Batch]:
        """
        generate val batch with pseudo labels
        :return:
        """
        val_batch = self.get_batch_data(self.val_indices)
        val_x = val_batch.X.astype(float)
        pseudo_s = sample_n_k(len(val_x), self.num_classes)
        pseudo_q = np.ones(len(val_x), dtype=bool)
        pseudo_q[pseudo_s] = False
        # rand_embed = val_x @ np.random.uniform(size=(len(self.all_features), 10))
        _, pseudo_labels = jit_soft_kmeans(val_x,  # (B, embed_dim), raw vector as embeddings
                                           val_x[pseudo_s],  # (n_way, embed_dim)
                                           adaptive_steps=5,
                                           temperature=1,
                                           )

        support = Batch(X=val_x[pseudo_s],
                        num_idx=self.num_idx,
                        cate_idx=self.cat_idx,
                        # cate_mask=val_batch.cate_mask[pseudo_s],
                        y=np.arange(self.num_classes))
        query = Batch(X=val_x[pseudo_q],
                      num_idx=self.num_idx,
                      cate_idx=self.cat_idx,
                      # cate_mask=val_batch.cate_mask[pseudo_q],
                      y=pseudo_labels[pseudo_q])

        return support, query

    def sample_episode(self, n_way, n_shot, n_query):
        classes = np.random.choice(self.labels, size=n_way, replace=False)
        classes.sort()  # Ensure classes are sorted
        support_data, support_labels, query_data, query_labels = [], [], [], []

        for cls in classes:
            cls_indices = np.where(self.y == cls)[0]
            np.random.shuffle(cls_indices)

            support_indices = cls_indices[:n_shot]
            query_indices = cls_indices[n_shot:n_shot + n_query]

            support_data.extend(self.get_batch_data(support_indices).X)
            support_labels.extend([cls]*n_shot)

            query_data.extend(self.get_batch_data(query_indices).X)
            query_labels.extend([cls]*n_query)

        return (
            np.array(support_data),
            np.array(support_labels),
            np.array(query_data),
            np.array(query_labels)
        )

    # def sample_episode(self, n_way, n_shot, n_query):
    #     classes = np.random.choice(self.labels, size=n_way, replace=False)
    #     support_data, support_labels, query_data, query_labels = [], [], [], []

    #     for cls in classes:
    #         cls_indices = np.where(self.y == cls)[0]
    #         np.random.shuffle(cls_indices)

    #         support_indices = cls_indices[:n_shot]
    #         query_indices = cls_indices[n_shot:n_shot + n_query]

    #         # Fetch data for support and query
    #         if self.num_features:
    #             support_data.extend(self.X_num.iloc[support_indices].to_numpy())
    #             query_data.extend(self.X_num.iloc[query_indices].to_numpy())
    #         if self.cat_features:
    #             support_data.extend(self.X_cat.iloc[support_indices].to_numpy())
    #             query_data.extend(self.X_cat.iloc[query_indices].to_numpy())

    #         support_labels.extend([cls]*n_shot)
    #         query_labels.extend([cls]*n_query)

    #     #return support_data, support_labels, query_data, query_labels
    #     return (
    #     np.array(support_data), 
    #     np.array(support_labels),
    #     np.array(query_data), 
    #     np.array(query_labels)
    #     )
    
    # def sample_episode(self, n_way, n_shot, n_query):
    #     classes = np.random.choice(self.labels, size=n_way, replace=False)
    #     support_data, support_labels, query_data, query_labels = [], [], [], []

    #     for cls in classes:
    #         cls_indices = np.where(self.y == cls)[0]
    #         np.random.shuffle(cls_indices)
    #         support_indices = cls_indices[:n_shot]
    #         query_indices = cls_indices[n_shot:n_shot + n_query]

    #         support_data.extend([self.get_data_point(index) for index in support_indices])
    #         support_labels.extend([cls] * n_shot)
    #         query_data.extend([self.get_data_point(index) for index in query_indices])
    #         query_labels.extend([cls] * n_query)

    #     return (support_data, support_labels), (query_data, query_labels)

    # def sample_episode(self, n_way, n_shot, n_query):
    #     """
    #     Sample an episode for few-shot learning, consisting of n_way classes,
    #     with n_shot examples per class for the support set, and n_query examples per class for the query set.
    #     """
    #     assert self.train_indices is not None, "Training indices should be defined."

    #     classes = np.random.choice(self.labels, size=n_way, replace=False)  # Ensure that classes are unique
    #     support_set = []
    #     query_set = []
        
    #     for cls in classes:
    #         # Find indices of all examples belonging to the class `cls`
    #         cls_indices = np.where(self.y[self.train_indices] == cls)[0]
    #         cls_indices = self.train_indices[cls_indices]
    #         assert len(cls_indices) >= (n_shot + n_query), f"Not enough samples for class {cls} to form an episode."

    #         # Shuffle indices to randomize selection
    #         np.random.shuffle(cls_indices)
            
    #         # Select support and query indices
    #         support_indices = cls_indices[:n_shot]
    #         query_indices = cls_indices[n_shot:n_shot + n_query]

    #         # Extract support and query samples
    #         support_set.extend(support_indices)
    #         query_set.extend(query_indices)
        
    #     # Convert indices to actual data samples
    #     support_batch = self.get_batch_data(np.array(support_set))
    #     query_batch = self.get_batch_data(np.array(query_set))

    #     return support_batch, query_batch

    def get_test_batch(self) -> Batch:
        return self.get_batch_data(self.test_indices)

    def get_training_batch(self) -> Batch:
        return self.get_batch_data(self.train_indices)

    def get_batch_data(self, indices) -> Union[Batch, pd.DataFrame]:
        numerical_feature = self.X_num.loc[indices].to_numpy() if self.num_features else None
        categorical_feature = self.X_cat.loc[indices].to_numpy() if self.cat_features else None

        if numerical_feature is None:
            features = categorical_feature
        else:
            features = numerical_feature
            if categorical_feature is not None:
                features = np.concatenate([features, categorical_feature], axis=-1)

        return Batch(X=features,  # concat(num, cat)
                     num_idx=self.num_idx,
                     cate_idx=self.cat_idx,
                     y=self.y[indices].to_numpy())

    def summary(self):
        num_features = ''.join([n + '\n ' if i % 5 == 4 else n + ' ' for i, n in enumerate(self.num_features)])
        class_prop = list(round(_, 4) for _ in self.y.value_counts()/len(self.y))
        summary = "######### Dataset information ######### \n"
        summary += f'Dataset={self.data_name}, train={len(self.train_indices)}, test={len(self.test_indices)}'
        summary += f'\n#(Classes)={self.num_classes} proportion={class_prop}'
        summary += f'\n#(Numerical features)={len(self.num_features)}:\n' + num_features
        summary += f'\n#(Categorical features)={len(self.cat_features)}:'
        summary += '\n\t'.join([' '] + [' x'.join([i, str(j)]) for i, j in zip(self.cat_features, self.cat_num)])
        print(summary)


if __name__ == '__main__':
    # dataset = openml.datasets.get_dataset('dna')
    # Xy, _, _, _ = dataset.get_data(dataset_format="dataframe")

    self = OpenmlDataset('jungle', one_hot=True)

    # self.y.value_counts()/len(self.y)

