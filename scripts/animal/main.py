import os
import numpy as np
import sacred
import torch
import torch.utils.data
import torch.utils.data.distributed
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from tools.trainer import trainer_ingredient
from tools.eval import Evaluator
from tools.eval import eval_ingredient
from tools.ugda import ugda_ingredient
from tools.optim import optim_ingredient
from datasets.ingredient import dataset_ingredient
from models.ingredient import model_ingredient
from tools.utils import save_pickle, load_pickle
torch.multiprocessing.set_sharing_strategy('file_system')

ex = sacred.Experiment('Scripts for FPML',
                                        ingredients=
                                        [trainer_ingredient,
                                        eval_ingredient,
                                        optim_ingredient,
                                        dataset_ingredient,
                                        model_ingredient,
                                        ugda_ingredient],
                                        save_git_info=False)
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    disable_tqdm = False
    cuda = True
    Notes=None
    Project=None
    dis_dir = None
    test_dir = None
    task_dir = None
    base_dir = None
    gaussian_dir = None
    gpu_id = None
    gpu_shift=None


@ex.automain
def main(
         gpu_id,
         gpu_shift,
         cuda,
         disable_tqdm,
         dis_dir,
         test_dir,
         task_dir,
         gaussian_dir,
         base_dir,
         Notes,
         Project
         ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_shift+gpu_id)
    device = torch.device("cuda" if cuda else "cpu", gpu_id)
    callback = None
    modal_index = ['0', '1']
    print("{} --> {}".format(Notes, Project))
    if not os.path.exists(dis_dir): os.makedirs(dis_dir)

    base_dict = load_pickle(os.path.join(base_dir, "base.plk"))
    base_classes = np.unique(base_dict['label'])
    anchors = {}

    for modal in modal_index:
        base_feats_ = base_dict['sample'][modal]
        base_labels_ = base_dict['label']
        feat_dim = base_feats_.shape[-1]
        mean = np.zeros([len(base_classes), feat_dim])
        cov = np.zeros([len(base_classes), feat_dim, feat_dim])
        for c in base_classes:
            feats_ = base_feats_[base_labels_ == c]
            idx_ = np.where(base_classes == c)[0]
            mean[idx_] = feats_.mean(0)
            cov[idx_] = np.cov(feats_.T)
        anchors[modal] = {"mean": mean, "cov": cov}
        save_pickle(os.path.join(dis_dir,
                                 "dis_{}.plk".format(modal)), anchors[modal])

    # >>>> Evaluation <<<<#
    test_path = os.path.join(test_dir, "test.plk")
    evaluator = Evaluator(device=device, ex=ex)
    results = evaluator.eval_protocol(modal_index=modal_index,
                                                  dis_dir=dis_dir,
                                                  test_path=test_path,
                                                  task_dir=task_dir,
                                                  gaussian_dir=gaussian_dir,
                                                  callback=callback,
                                                  disable_tqdm=disable_tqdm,
                                                  wandb=None)
    return results