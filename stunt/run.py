import sys

import torch

import time
# from common.utils import get_optimizer, load_model
# from data.dataset import get_meta_dataset
# from models.model import get_model
import numpy as np
import torch
from torch import nn
from model import MLPProto
from utils import Logger, set_random_seed, MetricLogger, save_checkpoint, save_checkpoint_step
from argparse import ArgumentParser

"""Command-line argument parser for train."""

parser = ArgumentParser(
    description='PyTorch implementation of Self-generated Tasks from UNlabeled Tables (STUNT)'
)

parser.add_argument('--dataset', help='Dataset', default='income', type=str)
parser.add_argument('--mode', help='Training mode', default='protonet', type=str)
parser.add_argument("--seed", type=int, default=0, help='random seed')
parser.add_argument("--rank", type=int, default=0, help='Local rank for distributed learning')
parser.add_argument('--distributed', help='automatically change to True for GPUs > 1', default=False, type=bool)
parser.add_argument('--resume_path', help='Path to the resume checkpoint', default=None, type=str)
parser.add_argument('--load_path', help='Path to the loading checkpoint', default=None, type=str)
parser.add_argument("--no_strict", help='Do not strictly load state_dicts', action='store_true')
parser.add_argument('--suffix', help='Suffix for the log dir', default=None, type=str)
parser.add_argument('--eval_step', help='Epoch steps to compute accuracy/error', default=50, type=int)
parser.add_argument('--save_step', help='Epoch steps to save checkpoint', default=2500, type=int)
parser.add_argument('--print_step', help='Epoch steps to print/track training stat', default=50, type=int)
parser.add_argument("--regression", help='Use MSE loss (automatically turns to true for regression tasks)',
                    action='store_true')
parser.add_argument("--baseline", help='do not save the date', action='store_true')

""" Training Configurations """
parser.add_argument('--outer_steps', help='meta-learning outer-step', default=10000, type=int)
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (absolute lr)')
parser.add_argument('--batch_size', help='Batch size', default=4, type=int)
parser.add_argument('--test_batch_size', help='Batch size for test loader', default=4, type=int)
parser.add_argument('--max_test_task', help='Max number of task for inference', default=1000, type=int)

""" Meta Learning Configurations """
parser.add_argument('--num_ways', help='N ways', default=10, type=int)
parser.add_argument('--num_shots', help='K (support) shot', default=1, type=int)
parser.add_argument('--num_shots_test', help='query shot', default=15, type=int)
parser.add_argument('--num_shots_global', help='global (or distill) shot', default=0, type=int)

""" Classifier Configurations """
parser.add_argument('--model', help='model type', type=str, default='mlp')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

P = parser.parse_args()


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def check(P):
    filename_with_today_date = True
    assert P.num_shots_global == 0
    return filename_with_today_date


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
                              - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()) * 100.


def train_func(P, step, model, criterion, optimizer, batch, metric_logger, logger):
    stime = time.time()
    model.train()

    assert not P.regression

    train_inputs, train_targets = batch['train']
    num_ways = len(set(list(train_targets[0].numpy())))
    test_inputs, test_targets = batch['test']

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)

    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
    loss = criterion(-squared_distances, test_targets)

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime)
    metric_logger.meters['meta_test_cls'].update(loss.item())
    metric_logger.meters['train_acc'].update(acc)

    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTestLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_test_cls.global_avg))


def test_func(P, model, loader, criterion, steps, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, batch in enumerate(loader):
        if n * P.test_batch_size > P.max_test_task:
            break

        train_inputs, train_targets = batch['train']

        num_ways = len(set(list(train_targets[0].numpy())))
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        with torch.no_grad():
            train_embeddings = model(train_inputs)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)
        with torch.no_grad():
            test_embeddings = model(test_inputs)

        prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                       - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
        loss = criterion(-squared_distances, test_targets)

        acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

        metric_logger.meters['loss'].update(loss.item())
        metric_logger.meters['acc'].update(acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [Acc@1 %.3f] [LossOut %.3f]' %
         (metric_logger.acc.global_avg, metric_logger.loss.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/acc', metric_logger.acc.global_avg, steps)
        logger.scalar_summary('eval/loss_test', metric_logger.loss.global_avg, steps)

    model.train(mode)

    return metric_logger.acc.global_avg


def train_setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}_{P.num_ways}way_{P.num_shots}shot_{P.num_shots_test}query'

    if mode == 'protonet':
        assert not P.regression

    else:
        raise NotImplementedError()

    today = check(P)
    if P.baseline:
        today = False

    # fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train_func, fname, today


# def test_setup(mode):
#     if mode == 'protonet':
# from evals.metric_based.protonet import test_classifier as test_func

# return test_func


def meta_trainer(train_func, test_func, model, optimizer, train_loader, test_loader, logger):
    kwargs = {}
    kwargs_test = {}

    metric_logger = MetricLogger(delimiter="  ")

    """ resume option """
    # is_best, start_step, best, acc = is_resume(P, model, optimizer)

    """ define loss function """
    criterion = nn.CrossEntropyLoss()
    best = 0
    """ training start """
    logger.log_dirname(f"Start training")
    for step in range(0, P.outer_steps + 1):

        stime = time.time()
        train_batch = next(train_loader)
        metric_logger.meters['data_time'].update(time.time() - stime)

        train_func(P, step, model, criterion, optimizer, train_batch,
                   metric_logger=metric_logger, logger=logger, **kwargs)

        """ evaluation & save the best model """
        if step % P.eval_step == 0:
            acc = test_func(P, model, test_loader, criterion, step, logger=logger, **kwargs_test)
            if best < acc:
                best = acc
                selected_acc = test_income(model)  # TODO: test on the true datset
                print(f"[Step={step}]Test acc = {selected_acc}")
                # save_checkpoint(P, step, best, model.state_dict(),
                #                 optimizer.state_dict(), logger.logdir, is_best=True)

            logger.scalar_summary('eval/best_acc', best, step)
            logger.log('[EVAL] [Step %3d] [Acc %5.2f] [Best %5.2f] [Test %5.2f]' % (step, acc, best, selected_acc))

        """ save model per save_step steps"""
        if step % P.save_step == 0:
            save_checkpoint_step(P, step, best, model.state_dict(),
                                 optimizer.state_dict(), logger.logdir)

    """ save last model"""
    save_checkpoint(P, P.outer_steps, best, model.state_dict(),
                    optimizer.state_dict(), logger.logdir)


def get_meta_dataset(dataset):
    from datasets import Income
    if dataset == 'income':
        meta_train_dataset = Income(tabular_size=105,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way=P.num_ways,
                                    query=P.num_shots_test)

        meta_val_dataset = Income(tabular_size=105,
                                  seed=P.seed,
                                  source='val',
                                  shot=1,
                                  tasks_per_batch=P.test_batch_size,
                                  test_num_way=2,
                                  query=30)

    else:
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset


def test_income(model):
    output_size = 2

    train_x = np.load('./data/income/xtrain.npy')
    train_y = np.load('./data/income/ytrain.npy')
    test_x = np.load('./data/income/xtest.npy')
    test_y = np.load('./data/income/ytest.npy')
    train_idx = np.load('./data/income/index{}/train_idx_{}.npy'.format(P.num_shots, P.seed))


    # train_x = np.load('./data/' + P.dataset + '/xtrain.npy')
    # train_y = np.load('./data/' + P.dataset + '/ytrain.npy')
    # test_x = np.load('./data/' + P.dataset + '/xtest.npy')
    # test_y = np.load('./data/' + P.dataset + '/ytest.npy')
    # train_idx = np.load('./data/' + P.dataset + '/index{}/train_idx_{}.npy'.format(P.num_shots, P.seed))

    few_train = model(torch.tensor(train_x[train_idx]).float())
    support_x = few_train.detach().numpy()
    support_y = train_y[train_idx]
    few_test = model(torch.tensor(test_x).float())
    query_x = few_test.detach().numpy()
    query_y = test_y

    def get_accuracy(prototypes, embeddings, targets):
        sq_distances = torch.sum((prototypes.unsqueeze(1)
                                  - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        return torch.mean(predictions.eq(targets).float()) * 100.

    train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
    train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
    val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    prototypes = get_prototypes(train_x, train_y, output_size)
    return get_accuracy(prototypes, val_x, val_y).item()

    # print(P.seed, acc)

    # out_file = 'result/{}_{}shot/test'.format(P.dataset, P.num_shots)
    # # with open(out_file, 'a+') as f:
    # #     f.write('seed: ' + str(P.seed) + ' test: ' + str(acc))
    # #     f.write('\n')


def main():
    P.rank = 0

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset and dataloader """
    # kwargs = {'batch_size': P.batch_size, 'shuffle': True,
    #           'pin_memory': True, 'num_workers': 2}
    train_set, val_set = get_meta_dataset(dataset=P.dataset)

    train_loader = train_set
    test_loader = val_set

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    model = MLPProto(105, 1024, 1024).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=P.lr)

    # optimizer = optim.Adam(params, lr=P.lr)
    # optimizer = get_optimizer(P, model)

    """ define train and test type """
    # from train import setup as train_setup
    # from evals import setup as test_setup
    # train_func, fname, today = train_setup(P.mode, P)
    # test_func = test_setup(P.mode)

    """ define logger """
    fname = f'{P.dataset}_{P.model}_{P.mode}_{P.num_ways}way_{P.num_shots}shot_{P.num_shots_test}query'
    today = True
    logger = Logger(fname, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ load model if necessary """
    # load_model(P, model, logger)

    """ train """
    meta_trainer(train_func, test_func, model, optimizer, train_loader, test_loader, logger)

    """ close tensorboard """
    logger.close_writer()


if __name__ == '__main__':
    main()
