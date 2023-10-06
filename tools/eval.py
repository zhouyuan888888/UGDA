import pdb
from sacred import Ingredient
from tools.utils import warp_tqdm, \
                                                compute_confidence_interval
from tools.utils import load_pickle, save_pickle
import os
import torch
from tools.ugda import UGDA
from tools.get_sn import get_Sn
import numpy as np

eval_ingredient = Ingredient('eval')
@eval_ingredient.config
def config():
    refresh_task = False
    refresh_gaussian = False
    tukey_trans = None
    #gaussian_dir = None
    target_data_path = None
    target_split_dir = None
    missing_rate = None
    task_num = 400
    way = 3
    shot = 1
    query_num = 15
    latent_dim = 512
    sampled_num = 5
    beta = 1e-3
    topk = 2
    batch_num = 50

class Evaluator:
    @eval_ingredient.capture
    def __init__(
                    self,
                    device,
                    ex
                    ):
        self.device = device
        self.ex = ex

    @eval_ingredient.capture
    def eval_protocol(
                                        self,
                                        modal_index,
                                        dis_dir,
                                        test_path,
                                        task_dir,
                                        gaussian_dir,
                                        task_num,
                                        way,
                                        shot,
                                        missing_rate,
                                        latent_dim,
                                        refresh_task,
                                        refresh_gaussian,
                                        tukey_trans, beta,
                                        sampled_num,
                                        topk,
                                        batch_num,
                                        wandb,
                                        disable_tqdm,
                                        callback,
                                        ):

        results = list()
        test_sample = load_pickle(test_path)

        sampled_num = int(sampled_num/shot)
        tasks = self.task_generator(
                                        test_sample=test_sample,
                                        shot=shot,
                                        task_dir=task_dir,
                                        missing_rate=missing_rate,
                                        refresh_task=refresh_task,
                                        disable_tqdm=disable_tqdm
                                        )

        logs = self.run_task(
                                    modal_index=modal_index,
                                    refresh_gaussian=refresh_gaussian,
                                    tukey_trans=tukey_trans,
                                    tasks=tasks,
                                    shot=shot,
                                    missing_rate=missing_rate,
                                    task_num=task_num,
                                    latent_dim=latent_dim,
                                    beta=beta,
                                    sampled_num=sampled_num,
                                    dis_dir=dis_dir,
                                    wandb=wandb,
                                    gaussian_dir=gaussian_dir,
                                    topk=topk,
                                    batch_num=batch_num,
                                    callback=callback,
                                    disable_tqdm=disable_tqdm
                                    )

        l2n_mean, l2n_conf = \
            compute_confidence_interval(logs)

        text = "missing_rate{}==> " \
                     "{}-way {}-shot: mean={}, std={}"\
                    .format(missing_rate,
                                    way,
                                    shot,
                                    l2n_mean,
                                    l2n_conf)
        results.append([text])
        return results


    def run_task(
                self,
                modal_index,
                refresh_gaussian,
                tukey_trans,
                tasks,
                shot,
                missing_rate,
                task_num,
                latent_dim,
                beta,
                sampled_num,
                dis_dir,
                gaussian_dir,
                topk,
                batch_num,
                wandb,
                callback,
                disable_tqdm
                ):

        builder = self.builder()
        x_s= tasks['x_s']
        y_s = tasks['y_s']
        x_q = tasks['x_q']
        y_q = tasks['y_q']

        # >>>> Data in numpy array <<<< #
        supports = [np.power(v, tukey_trans)
                                                for k, v in x_s.items()]
        queries = [np.power(v, tukey_trans)
                                                for k,v in x_q.items()]
        mean = dict()
        cov = dict()

        for idx_ in modal_index:
            filepath=os.path.join(dis_dir,
                                  'dis_{}.plk'.format(idx_))
            loading = load_pickle(filepath)
            mean[idx_] = loading['mean']
            cov[idx_] = loading['cov']

        mean = [v for k, v in mean.items()]
        cov = [v for k, v in cov.items()]

        n_task = task_num
        n_view = len(supports)
        n_batch = batch_num
        n_s = supports[0].shape[1]
        n_q = queries[0].shape[1]
        n_latent_dim = latent_dim
        n_view_dims = [supports[i].shape[-1]
                                for i in range(n_view)]

        # >>>> View missing indicator generation <<<<#
        indicators = list()
        for i in range(n_task):
            indicator_ = get_Sn(n_view, (n_s+n_q),
                                missing_rate=missing_rate)
            indicators.append(indicator_[np.newaxis, :, :])
        indicators = np.concatenate(indicators, axis=0)
        indicators_s = indicators[:, :n_s, :]
        indicators_q = indicators[:, n_s:, :]
        missing_s = 1-indicators_s.sum()\
                                    /(n_task*n_s*n_view)
        missing_q= 1 - indicators_q.sum() \
                                    / (n_task * n_q * n_view)
        print(">>>>>>missing rate of supports/queries: "
                    "{}/{}".format(missing_s,missing_q))

        # >>>>  filtering the information
        # in missing views <<<< #
        for n in range(n_view):
            indicators_s_ = indicators_s[:, :, n]
            indicators_q_ = indicators_q[:, :, n]
            supports[n] = supports[n] \
                          * indicators_s_[:, :, np.newaxis]
            queries[n] = queries[n] \
                          * indicators_q_[:, :, np.newaxis]

        # >>>>> generate gaussian nodes <<<<<<
        anchors = builder.Gaussian(
                                        supports=supports,
                                        support_labels=y_s,
                                        queries=queries,
                                        query_labels=y_q,
                                        indicators=indicators,
                                        mean=mean,
                                        cov=cov,
                                        gaussian_dir=gaussian_dir,
                                        shot=shot,
                                        n_view=n_view,
                                        missing_rate=missing_rate,
                                        n_task=n_task,
                                        n_s=n_s,
                                        n_q=n_q,
                                        refresh_gaussian=refresh_gaussian,
                                        device=self.device,
                                        beta=beta,
                                        sampled_num=sampled_num,
                                        topk=topk
                                        )

        n_s = n_s * sampled_num
        labels = anchors['label']
        anchors = anchors["anchor"]
        torch.cuda.empty_cache()

        task_loader = warp_tqdm(
                                        range(int(n_task/n_batch)),
                                        disable_tqdm=disable_tqdm)
        logs = list()
        for t in task_loader:
            if disable_tqdm is not True:
                task_loader.set_description(
                                            "load batch data >>>")
            samples_ = [V[t_*b_:(t_+1)*b_ ,:, :]
                                    for V, t_, b_ in zip(
                                                                anchors,
                                                                [t]*n_view,
                                                                [n_batch]*n_view
                                                                )]
            labels_=labels[t*n_batch:(t+1)*n_batch, :]
            indicators_ = indicators[t*n_batch:(t+1)*n_batch, :]
            builder.init_info_lists()

            builder.calculate_weights(
                                                n_s=n_s,
                                                n_q=n_q,
                                                y_s=y_s
                                                    )

            builder.latent(
                                n_batch=n_batch,
                                n_s=n_s,
                                n_q=n_q,
                                n_latent_dim=n_latent_dim,
                                device=self.device,
                                )

            builder.aggregator(
                                        n_view_dims=n_view_dims,
                                        n_latent_dim=n_latent_dim,
                                        device=self.device
                                        )

            latents = builder.inverse_agg(
                                                        samples=samples_,
                                                        labels=labels_,
                                                        indicators=indicators_,
                                                        n_s=n_s,
                                                        n_view=n_view,
                                                        device=self.device,
                                                        wandb=wandb
                                                        )

            builder.prototyping(
                                        latents=latents,
                                        labels=labels_,
                                        n_s=n_s,
                                        device=self.device
                                        )

            builder.calibration(
                                    latents=latents,
                                    labels=labels_,
                                    n_s=n_s,
                                    callback=callback,
                                    device=self.device
                                    )

            acc = builder.get_logs()['acc'][:, -1]
            logs.extend(list(acc))
        return logs

    @eval_ingredient.capture
    def builder(self):
        return UGDA()

    @eval_ingredient.capture
    def get_loaders(
                                self,
                                used_set,
                                target_data_path,
                                target_split_dir
                                ):

        # First, get loaders
        loaders_dic = {}
        loader_info = {
                                'aug': False,
                                'shuffle': False,
                                'out_name': False
                                }

        if target_data_path is not None:
            loader_info.update({
                                        'path': target_data_path,
                                        'split_dir': target_split_dir
                                                })

        train_loader = get_dataloader('train', **loader_info)
        loaders_dic['train_loader'] = train_loader

        test_loader = get_dataloader(used_set, **loader_info)
        loaders_dic.update({'test': test_loader})
        return loaders_dic

    @eval_ingredient.capture
    def extract_features(
                                    self,
                                    model,
                                    ckpt_path,
                                    model_tag,
                                    used_set,
                                    fresh_start,
                                    loaders_dic,
                                    disable_tqdm
                                    ):

        for i in range(1, len(ckpt_path)):
           assert "/".join(ckpt_path[i].split("/")[:-1]) == \
                  "/".join(ckpt_path[0].split("/")[:-1])

        save_dir = os.path.join("/".join(
            ckpt_path[0].split("/")[:-1]), model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)

            print("Loading features from {}".format(filepath))
            return extracted_features_dic
        # ... otherwise just extract them
        else:
            print("Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        for m in model:
            m.eval()

        with torch.no_grad():
            all_features = {}
            all_labels = []

            for i in range(len(ckpt_path)):
                all_features[ckpt_path[i].split("/")[-1]]=list()

            for i, (inputs, labels, _) in enumerate(
                                                    warp_tqdm(loaders_dic['test'],
                                                      disable_tqdm=disable_tqdm)):
                inputs = inputs.to(self.device)
                for j in range(len(model)):
                    outputs, _ = model[j](inputs, True)
                    all_features[ckpt_path[j].split("/")[-1]]\
                        .append(outputs.cpu())
                all_labels.append(labels)

            all_labels = torch.cat(all_labels, 0)
            for k in all_features.keys():
                all_features[k] = torch.cat(all_features[k], 0)

            extracted_features_dic = {
                                      'concat_features': all_features,
                                      'concat_labels': all_labels
                                                            }

        print("Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    @eval_ingredient.capture
    def get_task(
                        self,
                        shot,
                        way,
                        query_num,
                        extracted_features_dic
                        ):

        all_features = extracted_features_dic['sample']
        keys = [k for k in all_features.keys()]
        labels = extracted_features_dic['label']
        classes = np.unique(labels)

        sampled_classes = np.random.choice(
                                                                       a=classes,
                                                                       size=way,
                                                                       replace=False
                                                                        )

        supports = dict()
        queries = dict()
        for k in keys:
            supports[k]=list()
            queries[k]=list()

        for c in sampled_classes:
            indexes_ = np.where(labels == c)[0]
            indexes = np.random.choice(
                                        a=indexes_,
                                       size=shot + query_num,
                                        replace=False
                                        )
            for k in keys:
                supports[k].append(all_features[k][indexes[:shot]])
                queries[k].append(all_features[k][indexes[shot:]])

        x_supports = {k: np.concatenate(supports[k], axis=0)
                                                                                            for k in keys}
        x_queries = {k: np.concatenate(queries[k], axis=0)
                                                                                            for k in keys}

        y_supports = np.arange(way)[:, np.newaxis]\
            .repeat(repeats=shot, axis=1).reshape(-1)
        y_queries = np.arange(way)[:, np.newaxis]\
            .repeat(repeats=query_num, axis=1).reshape(-1)

        task = {'x_s': x_supports,
                        'y_s': y_supports,
                        'x_q': x_queries,
                        'y_q': y_queries}
        return task

    @eval_ingredient.capture
    def task_generator(
                                    self,
                                    test_sample,
                                    shot,
                                    task_num,
                                    task_dir,
                                    missing_rate,
                                    refresh_task,
                                    disable_tqdm
                                    ):

        dir_ = task_dir
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        dir_ = os.path.join(dir_,
                                "MissRate{}".format(missing_rate))
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        dir_ = os.path.join(dir_, '{}shot'.format(shot))
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        filepath = os.path.join(dir_, '{}tasks_{}shot.plk'
                                .format(task_num, shot))

        if not os.path.exists(filepath) or refresh_task is True:
            tasks_ = warp_tqdm(range(task_num),
                               disable_tqdm=disable_tqdm)
            tasks_dics = []
            for _ in tasks_:
                task_dic = self.get_task(shot=shot,
                                         extracted_features_dic=test_sample)
                tasks_dics.append(task_dic)
                if disable_tqdm is not True:
                    tasks_.set_description("Construct {} episodes"
                                           .format(task_num))

            #---Merging all episodic tasks into one dict---#
            keys_1st_level =[k for k in tasks_dics[0].keys()]
            keys_2st_level = [k
                        for k in tasks_dics[0][keys_1st_level[0]].keys()]
            n_tasks = len(tasks_dics)

            merged_tasks = {}
            for i in keys_1st_level:
                if i == "x_s" or i == "x_q":
                    merged_tasks[i]=dict()
                    for j in keys_2st_level:
                        list_ = list()
                        for k in range(n_tasks):
                            list_.append(
                                tasks_dics[k][i][j][np.newaxis, :, :])
                        merged_tasks[i][j] = np.concatenate(
                                                                                list_, axis=0)
                elif i == "y_s" or i == "y_q":
                        merged_tasks[i] = np.concatenate(
                                            [tasks_dics[j][i][np.newaxis, :]
                                             for j in range(n_tasks)], axis=0)
        else:
            print("Loading {} tasks from {}"
                  .format(task_num, filepath))
            merged_tasks = load_pickle(filepath)
        return merged_tasks
