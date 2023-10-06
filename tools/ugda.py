import pdb

from .utils import get_mi, get_cond_entropy, get_entropy, get_one_hot
from sacred import Ingredient
import torch
import time
import numpy as np
from models.Enc import Enc
import os
from tools.utils import load_pickle, save_pickle
from tools.utils import warp_tqdm
import torch.nn as nn

ugda_ingredient = Ingredient('ugda')
@ugda_ingredient.config
def config():
    lr = None
    iter = None
    rec_lr =None
    rec_iter = None
    alpha = 1
    temp = 1
    loss_weights = [1, 1, 1]
    update_net_iter= 10
    update_latent_iter = 10

class UGDA(object):
    @ugda_ingredient.capture
    def __init__(self,
                         temp,
                         loss_weights,
                         iter,
                         rec_lr,
                         lr,
                         rec_iter,
                         update_net_iter,
                         update_latent_iter,
                         alpha
                         ):

        self.temp = temp
        self.alpha=alpha
        self.loss_weights = loss_weights.copy()
        self.iter = iter
        self.lr = lr

        self.rec_lr = rec_lr
        self.rec_iter = rec_iter
        self.update_net_iter = update_net_iter
        self.update_latent_iter = update_latent_iter
        self.init_info_lists()


    def xavier_init(
                            self,
                            fan_in,
                            fan_out,
                            constant=1
                            ):

        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        init_represents = np.random.uniform(low=low,
                                            high=high, size=(fan_in, fan_out))
        init_represents = torch.Tensor(init_represents)
        return init_represents

    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

    def get_logits(
                            self,
                            samples,
                            weights
                            ):

        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(weights.transpose(1, 2)) \
                              - 1 / 2 * (weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(
                            self,
                            samples
                            ):

        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def latent(
                    self,
                    n_batch,
                    n_s,
                    n_q,
                    n_latent_dim,
                    device,
                    ):

        self.latent_res = \
                        self.xavier_init(n_batch*(n_s+n_q), n_latent_dim)\
                        .view(n_batch, (n_s+n_q), n_latent_dim)\
                        .to(device=device)

    def aggregator(
                        self,
                        n_view_dims,
                        n_latent_dim,
                        device
                        ):

        self.nets = list()
        for v in n_view_dims:
            self.nets.append(Enc(n_latent_dim, v).to(device))

    def prototyping(
                            self,
                            latents,
                            labels,
                            n_s,
                            device
                            ):

        t0 = time.time()
        support_represents = latents[:, :n_s, :]
        support_labels = labels[:, :n_s]
        query_represents = latents[:, n_s: , :]
        query_labels = labels[:, n_s: ]

        one_hot = get_one_hot(torch.from_numpy(
            support_labels)).to(device)
        counts = one_hot.sum(1).unsqueeze(-1)
        weights = one_hot.transpose(1, 2).matmul(
            support_represents)
        self.weights = weights / counts
        self.record_info_by_latent(
                         new_time=time.time()-t0,
                         support=support_represents,
                         query=query_represents,
                         y_s=support_labels,
                         y_q=query_labels,
                        device=device
                        )
        weights = self.weights
        return weights

    def fund_top_base_index(
                                            self,
                                            query,
                                            base_mean,
                                            base_cov,
                                            top_k
                                            ):

        dist=np.linalg.norm(query[np.newaxis,:] - base_mean,
                                                                axis=-1, keepdims=False)
        top_index=np.argpartition(dist, top_k)[:top_k]
        return top_index

    def Gaussian(
                        self,
                        supports,
                        support_labels,
                        queries,
                        query_labels,
                        indicators,
                        mean,
                        cov,
                        gaussian_dir,
                        shot,
                        n_view,
                        missing_rate,
                        n_task,
                        n_s,
                        n_q,
                        refresh_gaussian,
                        sampled_num,
                        device,
                        beta=0.21,
                        topk=2
                        ):

        self.sampled_num = sampled_num
        n_sample = n_s + n_q
        labels = np.concatenate([support_labels,
                                 query_labels], axis=-1)
        if not os.path.exists(gaussian_dir): os.mkdir(gaussian_dir)
        dir_ = os.path.join(gaussian_dir,
                            "missingrate-{}".format(missing_rate))
        if not os.path.exists(dir_): os.mkdir(dir_)
        dir_ = os.path.join(dir_, 'shot-{}'.format(shot))
        if not os.path.exists(dir_): os.mkdir(dir_)
        TopBaseIndex_path = os.path.join(dir_,
                                         'top_base.plk'.format(shot, missing_rate))
        SampledData_path = os.path.join(dir_,
                                        'anchors.plk'.format(shot, missing_rate))

        #---Generating top base indexes---#
        if not os.path.exists(TopBaseIndex_path) or refresh_gaussian:
            all_index = list()
            task_loader = warp_tqdm(range(n_task), False)
            for t in task_loader:
                index_for_all_sample = list()
                for n in range(n_sample):
                    index_for_one_sample = list()
                    for v in range(n_view):
                        view_ind = indicators[t][n][v]
                        if view_ind != 0:
                            base_mean = mean[v]
                            base_cov = cov[v]
                            sample = np.concatenate([supports[v][t],
                                                     queries[v][t]], axis=0)[n]
                            top_index= self.fund_top_base_index(sample,
                                            base_mean, base_cov, top_k=topk)
                            index_for_one_sample.extend(top_index)
                    index_for_all_sample.append(np.unique(
                        np.array(index_for_one_sample)))
                all_index.append(index_for_all_sample)
                task_loader.set_description("Generating top "
                                            "base indexes")
            save_pickle(TopBaseIndex_path, all_index)
        else:
            print("Loading top base index from >>>>> {}"
                  .format(TopBaseIndex_path))
            all_index = load_pickle(TopBaseIndex_path)

        #---Generating sampled data---#
        if not os.path.exists(SampledData_path) or refresh_gaussian:
            all_sample = list()
            all_label = list()

            for v in range(n_view):
                mean_ = torch.Tensor(mean[v])
                cov_ = torch.Tensor(cov[v])
                sample_for_all_task = list()
                task_loader = warp_tqdm(range(n_task), False)
                for t in task_loader:
                    sample_for_one_task = list()
                    label_for_one_task = list()

                    for n in range(n_sample):
                        view_ind = indicators[t][n][v]
                        top_ind_ = all_index[t][n]
                        sample = np.concatenate([supports[v][t],
                                                 queries[v][t]], axis=0)[n]
                        sample = torch.Tensor(sample).to(device)
                        mean_topk = mean_[top_ind_].to(device)
                        cov_topk = cov_[top_ind_].to(device)

                        if view_ind != 0:
                            mean__ = torch.cat([mean_topk.mean(0,
                                                                keepdim=True),
                                                                sample.unsqueeze(0)],
                                                                dim=0).mean(0)
                        else:
                            mean__ = mean_topk.mean(0)
                        cov__ = cov_topk.mean(0)

                        if n<n_s:
                            # ---> https://juanitorduz.github.io/
                            # multivariate_normal/
                            cov__ +=  beta * torch.eye(cov__.shape[0]).cuda()
                            dis = torch.distributions.MultivariateNormal(
                                        loc=mean__, covariance_matrix=cov__)
                            anchors = dis.sample([int(self.sampled_num)])
                            sample_for_one_task.extend(anchors[:, np.newaxis, :])
                            if v == 0:
                                label_for_one_task.extend(np.array([
                                        labels[t][n]] * int(self.sampled_num)))
                        else:
                            if view_ind != 0:
                                sampled_ = sample.unsqueeze(0)
                            else:
                                sampled_ = mean__.unsqueeze(0)
                            if v == 0:
                                label_for_one_task.extend(
                                                np.array([labels[t][n]]))
                            sample_for_one_task.extend(
                                                sampled_[:, np.newaxis, :])
                        torch.cuda.empty_cache()

                    sample_for_one_task = torch.cat(
                                    sample_for_one_task, dim=0)
                    sample_for_all_task.append(
                                    sample_for_one_task.unsqueeze(0))
                    if v==0:
                        all_label.append(
                                np.array(label_for_one_task)[np.newaxis, :])
                    task_loader.set_description("Generating "
                                "Gaussian nodes of {}-th view".format(v))

                sample_for_all_task = torch.cat(sample_for_all_task,
                                                                            dim=0)
                all_sample.append(
                    sample_for_all_task.cpu().detach().numpy())
                torch.cuda.empty_cache()

            all_label = np.concatenate(all_label, axis=0)
            generated_data = {"label": all_label,
                                                "anchor": all_sample}
        else:
            print("Loading generated "
                  "Gaussian data from >>>>> "
                  "{}".format(SampledData_path))
            generated_data =  load_pickle(SampledData_path)
        return generated_data

    def calculate_weights(
                                    self,
                                    n_s,
                                    n_q,
                                    y_s
                                    ):

        self.N_s, self.N_q = n_s, n_q
        try:
            self.num_classes = np.unique(y_s).shape[0]
        except:
            self.num_classes = torch.unique(y_s).shape[0]
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) \
                                   * self.N_s / self.N_q

    def record_info_by_latent(
                                            self,
                                            new_time,
                                            support,
                                            query,
                                            y_s,
                                            y_q,
                                            device
                                            ):

        logits_q = self.get_logits(query, self.weights).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        self.timestamps.append(new_time)

        self.mutual_infos.append(get_mi(probs=q_probs))
        self.entropy.append(get_entropy(
                    probs=q_probs.detach()))
        self.cond_entropy.append(get_cond_entropy(
                            probs=q_probs.detach()))
        self.test_acc.append((preds_q ==
                    torch.from_numpy(y_q).to(device)).float()
                    .mean(1, keepdim=True))


    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc,
                                  dim=1).cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy,
                                      dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy,
                                 dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos,
                                      dim=1).cpu().numpy()

        return {'timestamps': self.timestamps,
                        'mutual_info': self.mutual_infos,
                        'entropy': self.entropy,
                        'cond_entropy': self.cond_entropy,
                        'acc': self.test_acc,
                        'losses': self.losses}

    # >>>> initialization <<<< #
    def parameter_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
        else:
            pass

    def inverse_agg(
                            self,
                            samples,
                            labels,
                            indicators,
                            n_s,
                            n_view,
                            device,
                            wandb
                            ):

        n_batch = samples[0].shape[0]
        n_s = n_s
        n_q = samples[0].shape[1] - n_s

        indicators_q =  torch.from_numpy(
                                        indicators[:, -n_q:, :]).float().to(device)
        indicators = torch.cat(
            [torch.ones([n_batch, n_s, n_view]).to(device),
             indicators_q], dim=1)
        samples = [torch.from_numpy(s).to(device)
                    for s in samples]

        self.net_optimizer = torch.optim.Adam([
            {'params': m.parameters()
             for m in self.nets}], self.rec_lr)
        self.latent_optimizer = torch.optim.Adam([
            {'params': self.latent_res}], lr=self.rec_lr)

        iter_loader = warp_tqdm(range(self.rec_iter), False)
        for i in iter_loader:
            iter_loader.set_description("anchor agg >>>")

            # ---network update--- #
            if self.update_net_iter != 0:
                [m.train() for m in self.nets]
                self.latent_res.requires_grad_(False)
                for j in range(self.update_net_iter):
                    net_loss = 0.
                    if i == 0:
                        for v in range(n_view):
                            self.nets[v].apply(self.parameter_init)
                    for v in range(n_view):
                        latents = self.nets[v](self.latent_res)[:, :n_s, :]
                        anchors_ = samples[v][:, :n_s, :]
                        indicators_ = indicators[:, :n_s, v].unsqueeze(-1)
                        net_loss = net_loss \
                                   +torch.sum(torch.pow((latents - anchors_), 2) \
                                    *indicators_, dim=-1).mean(0).mean(0) \
                                   +torch.sum(torch.pow((latents - anchors_), 2)  \
                                    *(1 - indicators_),
                                              dim=-1).mean(0).mean(0) * self.alpha
                    self.net_optimizer.zero_grad()
                    net_loss.backward()
                    self.net_optimizer.step()

            #---anchor update---#
            if self.update_net_iter != 0:
                [m.eval() for m in self.nets]
                self.latent_res.requires_grad_(True)
                for step_latent_update in range(self.update_latent_iter):
                    latent_loss = 0.
                    for v in range(n_view):
                        latents = self.nets[v](self.latent_res)[:, :n_s, :]
                        anchors_ = samples[v][:, :n_s, :]
                        indicators_ = indicators[:, :n_s, v].unsqueeze(-1)
                        latent_loss = latent_loss \
                                      + torch.sum(torch.pow((latents - anchors_), 2) \
                                      * indicators_, dim=-1).mean(0).mean(0) \
                                      + torch.sum(torch.pow((latents - anchors_), 2) \
                                      * (1 - indicators_),
                                                  dim=-1).mean(0).mean(0) * self.alpha
                    self.latent_optimizer.zero_grad()
                    latent_loss.backward()
                    self.latent_optimizer.step()

        #---query update---#
        [m.eval() for m in self.nets]
        self.latent_res.requires_grad_(True)
        iter_loader = warp_tqdm(range(self.rec_iter), False)
        for i in iter_loader:
            iter_loader.set_description("query agg >>>")
            latent_loss = 0.
            for v in range(n_view):
                latents = self.nets[v](self.latent_res)[:, n_s:, :]
                anchors_ = samples[v][:, n_s:, :]
                indicators_ = indicators[:, n_s:, v].unsqueeze(-1)
                latent_loss = latent_loss \
                              +torch.sum(torch.pow((latents - anchors_), 2)
                              *indicators_, dim=-1).mean(0).mean(0) \
                              +torch.sum(torch.pow((latents - anchors_), 2)
                              *(1 - indicators_),
                                dim=-1).mean(0).mean(0) * self.alpha
            self.latent_optimizer.zero_grad()
            latent_loss.backward()
            self.latent_optimizer.step()

        [m.eval() for m in self.nets]
        self.latent_res.requires_grad_(False)
        return self.latent_res

    def calibration(
                        self,
                        latents,
                        labels,
                        n_s,
                        callback,
                        device
                        ):

        t0 = time.time()
        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)

        support_represents = latents[:, :n_s, :]
        support_labels = labels[:, :n_s]
        query_represents = latents[:, n_s: , :]
        query_labels = labels[:, n_s: ]

        support_labels_one_hot = get_one_hot(
            torch.from_numpy(support_labels)).to(device)
        task_loader = warp_tqdm(range(self.iter), False)

        for i in task_loader:
            logits_s = self.get_logits(support_represents,
                                       self.weights)
            logits_q = self.get_logits(query_represents,
                                       self.weights)

            ce = - (support_labels_one_hot * torch.log(
                logits_s.softmax(2) + 1e-12)).sum(2).mean(1).mean(0)
            q_probs = logits_q.softmax(2)
            q_ent = - (q_probs.mean(1) * torch.log(
                                    q_probs.mean(1))).sum(1).mean(0)

            loss = self.loss_weights[0] * ce \
                                    - (self.loss_weights[1] * q_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            if callback is not None:
                P_q = self.get_logits(query_represents,
                                      self.weights).softmax(2).detach()
                prec = (P_q.argmax(2) ==
                                            query_labels).float().mean()
                callback.scalar('prec', i, prec, title='Precision')

            self.record_info_by_latent(
                            new_time=t1-t0,
                            support=support_represents,
                            query=query_represents,
                            y_s=support_labels,
                            y_q=query_labels,
                            device=device
                            )
            t0 = time.time()
            weights = self.weights
        return weights, q_probs.mean(1)
