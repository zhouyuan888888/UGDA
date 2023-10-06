import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
from numpy.random import shuffle
import math

def get_one_hot(y_s):
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def get_logs_path(
                        model_path,
                        method,
                        shot
                            ):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(
                        model,
                        samples
                        ):
    features, _ = model(samples, True)
    features = F.normalize(features.view(
                features.size(0), -1), dim=1)
    return features


def get_loss(
                logits_s,
                logits_q,
                labels_s,
                lambdaa
                ):
    Q = logits_q.softmax(2)
    y_s_one_hot = get_one_hot(labels_s)
    ce_sup = - (y_s_one_hot * torch.log(
                    logits_s.softmax(2) + 1e-12)).sum(2).mean(1)
    ent_q = get_entropy(Q)
    cond_ent_q = get_cond_entropy(Q)
    loss = - (ent_q - cond_ent_q) + lambdaa * ce_sup
    return loss


def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):
    q_ent = - (probs.mean(1) *
               torch.log(probs.mean(1) + 1e-12)
               ).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):
    q_cond_ent = - (probs *
                    torch.log(probs + 1e-12)
                    ).sum(2).mean(1, keepdim=True)
    return q_cond_ent


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery,
                         query: 1. - F.cosine_similarity(query[:, None, :],
                         gallery[None, :, :], dim=2),
        'euclidean': lambda gallery,
                                query: ((query[:, None, :] -
                                gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery,
                                query: torch.norm((query[:, None, :] -
                                gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery,
                               query: torch.norm((query[:, None, :] -
                               gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s "
        "%(filename)s:"
        "%(lineno)s] "
        "%(levelname)-8s "
        "%(message)s",
        datefmt='%Y-%m-%d %H:%M:%S', )

    logger = logging.getLogger('example')
    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath,
                                      mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger

def warp_tqdm(
                        data_loader,
                        disable_tqdm,
                        leave=False
                        ):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(
                        data_loader,
                        total=len(data_loader),
                        leave=leave,
                        dynamic_ncols=True)
    return tqdm_loader

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(
                                    state,
                                    is_best,
                                    filename='checkpoint.pth.tar',
                                    folder='result/default'
                                    ):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder
                        + '/' + filename,
                        folder +
                        '/model_best.pth.tar')


def load_checkpoint(
                                model,
                                ckpt_path,
                                type='best'
                                ):
    if type == 'best':
        for i in range(len(model)):
            #print("Loading model ==> {}".format( ckpt_path[i].split("/")[-1]))
            checkpoint = torch.load('{}/'
                                    'model_best.pth.tar'.format(ckpt_path[i]))
            #state_dict = checkpoint['state']
            #model_dict_load = model[i].state_dict()
            #model_dict_load.update(state_dict)
            #model[i].load_state_dict(model_dict_load)
            state_dict = checkpoint['state_dict']
            names = []
            for k, v in state_dict.items():
                names.append(k)
            model[i].load_state_dict(state_dict)
    elif type == 'last':
        for i in range(len(model)):
            checkpoint = torch.load('{}/'
                                    'checkpoint.pth.tar'.format(ckpt_path[i]))
            state_dict = checkpoint['state_dict']
            names = []
            for k, v in state_dict.items():
                names.append(k)
            model[i].load_state_dict(state_dict)
    else:
        assert False, 'type should be in ' \
                      '[best, or last], but got {}'.format(type)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP)
    across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    #pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    pm = (std / np.sqrt(a.shape[axis]))
    return m, pm

def read_mat_data(
                            str_name,
                            ratio,
                            Normal=1
                                    ):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1)
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    classes = max(labels)[0]
    all_length = 0
    for c_num in range(1, classes + 1):
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)
        labels_train.extend(labels[all_length + index][
                            0:math.floor(c_length * ratio)])
        labels_test.extend(labels[all_length + index][
                           math.floor(c_length * ratio):])
        X_train_temp = []
        X_test_temp = []
        for v_num in range(view_number):
            X_train_temp.append(X[v_num][0][0].transpose()[
                    all_length + index][0:math.floor(c_length * ratio)])
            X_test_temp.append(X[v_num][0][0].transpose()[
                    all_length + index][math.floor(c_length * ratio):])
        if c_num == 1:
            X_train = X_train_temp;
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.r_[X_train[v_num],
                                       X_train_temp[v_num]]
                X_test[v_num] = np.r_[X_test[v_num],
                                      X_test_temp[v_num]]
        all_length = all_length + c_length

    if (Normal == 1):
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num])
            X_test[v_num] = Normalize(X_test[v_num])

    traindata = DataSet(
                                        X_train,
                                        view_number,
                                        np.array(labels_train)
                                    )
    testdata = DataSet(
                                        X_test,
                                        view_number,
                                        np.array(labels_test)
                                    )
    return traindata, testdata, view_number

def  Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)

class DataSet(object):

    def __init__(
                        self,
                        data,
                        view_number,
                        labels
                        ):
        """
        Construct a DataSet.
        """
        self.data = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples