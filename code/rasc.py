from __future__ import print_function
import os
import argparse
import math
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from image_list import ImageList
from prepocess import *
from network import ResNet50Fc
import numpy as np
from utils import *
from lr_scheduler import *
from forever_data_iter import *
import sampler

import random


class RASC(object):
    def __init__(self, args):
        self.args = args

        self.T = self.args.cont_t

        self.define_dataset()
        self.define_loader()

        self.sampling_epochs = [0, 2, 4, 6, 8]
        # VisDA and MiniDomainNet
        # self.sampling_epochs = [0, 1, 2, 3, 4]
        self.budget = self.args.budget
        self.num_anno = math.ceil((self.budget / len(self.sampling_epochs)) * len(self.target_train_ds))

        if self.args.pre_train:
            self.define_model_source_only()
        else:
            self.define_cb_loader()
            self.define_model_jrw_train()
            self.emb_mb = None
            self.emb_mb_aug = None
            self.lbl_mb = None
            self.emb_mb_reg = None
            self.emb_mb_reg_aug = None
            self.mb_init_sum = 0
            self.mb_reg_init_sum = 0
            self.mb_init_max_iter = max(1, (len(self.source_train_ds) / (self.args.n_samples * self.args.class_num)))
            self.mb_reg_init_max_iter = max(1, (
                    len(self.target_train_ds) - (len(self.sampling_epochs) * self.num_anno)) / self.args.batch_size)
            self.mb_dequeue_point = []
            self.mb_reg_dequeue_point = []

        self.define_optim()
        self.scheduler = get_inv_lr_scheduler(self.optim, last_epoch=0)

    def define_dataset(self):
        self.source_train_ds = ImageList(self.args.source, transform=train_transform, strong_transform=strong_transform)
        self.target_train_ds = ImageList(self.args.target, transform=train_transform, strong_transform=strong_transform)
        self.target_test_ds = ImageList(self.args.target_val, transform=test_transform, test=True)
        self.labeled_target_ds = ImageList(transform=train_transform, strong_transform=strong_transform, empty=True)

    def define_loader(self):
        self.source_train_loader = DataLoader(self.source_train_ds, batch_size=self.args.batch_size, shuffle=True,
                                              drop_last=True, num_workers=2)
        self.target_train_loader = DataLoader(self.target_train_ds, batch_size=self.args.batch_size, shuffle=True,
                                              drop_last=True, num_workers=2)
        self.target_test_loader = DataLoader(self.target_test_ds, batch_size=self.args.batch_size, shuffle=False,
                                             drop_last=False, num_workers=2)
        self.labeled_target_loader = DataLoader(self.labeled_target_ds,
                                                batch_size=min(self.args.batch_size, len(self.labeled_target_ds)),
                                                shuffle=True, drop_last=True, num_workers=2)

    def define_cb_loader(self):
        self.src_cb_sampler = sampler.get_sampler({
            'dataset': self.source_train_ds,
            'num_categories': self.args.class_num,
            'n_samples': self.args.n_samples,
        })
        self.source_cb_loader = torch.utils.data.DataLoader(self.source_train_ds, batch_sampler=self.src_cb_sampler,
                                                            shuffle=False, num_workers=2)

    def define_model_source_only(self):
        self.model = ResNet50Fc(bottleneck_dim=512, class_num=self.args.class_num).cuda()
        self.inc = 2048

    def define_model_jrw_train(self):
        model = torch.load(osp.join(self.args.save_model_path, "pre_train_model.pth.tar"))
        self.model = model.cuda()
        self.inc = 2048

        self.momentum_model = ResNet50Fc(bottleneck_dim=512, class_num=self.args.class_num).cuda()
        for param_model, param_momentum_model in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_momentum_model.data.copy_(param_model.data)
            param_momentum_model.requires_grad = False

    @torch.no_grad()
    def update_momentum_model(self):
        for param_model, param_momentum_model in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_momentum_model.data = param_momentum_model.data * self.args.m + param_model.data * (1.0 - self.args.m)

    def define_optim(self):
        param_groups = self.model.trainable_parameters()
        bblr = self.args.lr * 0.1
        param_list = [{'params': param_groups[0], 'lr': bblr, 'initial_lr': bblr},
                      {'params': param_groups[1], 'initial_lr': self.args.lr}]
        self.optim = optim.SGD(param_list, lr=self.args.lr, momentum=0.9, nesterov=True,
                               weight_decay=self.args.weight_decay)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        loader_iter = iter(self.target_test_loader)
        with torch.no_grad():
            for batch_idx in range(len(self.target_test_loader)):
                data_t = next(loader_iter)
                data, target = data_t[0].cuda(), data_t[1].cuda()
                _, output = self.model(data, getemb=True)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.target_test_loader.dataset)

        print(
            'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct,
                                                                               len(self.target_test_loader.dataset),
                                                                               100. * correct / len(
                                                                                   self.target_test_loader.dataset)))

        return correct / len(self.target_test_loader.dataset)

    def pre_train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target, _, _) in enumerate(self.source_train_loader):
            data, target = data.cuda(), target.cuda()
            _, output = self.model(data, getemb=True)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.scheduler.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                              len(self.source_train_loader.dataset),
                                                                              100. * batch_idx / len(
                                                                                  self.source_train_loader),
                                                                              loss.item()))

    @torch.no_grad()
    def update_mb(self, epoch, step, s_cb_data, lt_cb_data):
        if epoch > 0 or (epoch == 0 and step > 0):
            self.update_momentum_model()

        s_img = s_cb_data[0].cuda()
        s_img_aug = s_cb_data[3].cuda()
        s_lbl = s_cb_data[1].cuda()
        embs_s, _ = self.momentum_model(s_img, getemb=True)
        embs_s = nn.functional.normalize(embs_s, dim=1)
        embs_s_aug, _ = self.momentum_model(s_img_aug, getemb=True)
        embs_s_aug = nn.functional.normalize(embs_s_aug, dim=1)

        lt_img = lt_cb_data[0].cuda()
        lt_img_aug = lt_cb_data[3].cuda()
        lt_plbl = lt_cb_data[1].cuda()
        embs_lt, _ = self.momentum_model(lt_img, getemb=True)
        embs_lt = nn.functional.normalize(embs_lt, dim=1)
        embs_lt_aug, _ = self.momentum_model(lt_img_aug, getemb=True)
        embs_lt_aug = nn.functional.normalize(embs_lt_aug, dim=1)

        if self.mb_init_sum == 0:
            self.emb_mb = torch.cat((embs_s, embs_lt))
            self.emb_mb_aug = torch.cat((embs_s_aug, embs_lt_aug))
            self.lbl_mb = torch.cat((s_lbl, lt_plbl))
            self.mb_dequeue_point.append(len(embs_s) + len(embs_lt))
            self.mb_init_sum = self.mb_init_sum + 1
        elif self.mb_init_sum > 0 and self.mb_init_sum < self.mb_init_max_iter:
            self.emb_mb = torch.cat((self.emb_mb, embs_s, embs_lt))
            self.emb_mb_aug = torch.cat((self.emb_mb_aug, embs_s_aug, embs_lt_aug))
            self.lbl_mb = torch.cat((self.lbl_mb, s_lbl, lt_plbl))
            self.mb_dequeue_point.append(len(embs_s) + len(embs_lt))
            self.mb_init_sum = self.mb_init_sum + 1
        else:
            self.emb_mb = torch.cat((self.emb_mb[self.mb_dequeue_point[0]:], embs_s, embs_lt))
            self.emb_mb_aug = torch.cat((self.emb_mb_aug[self.mb_dequeue_point[0]:], embs_s_aug, embs_lt_aug))
            self.lbl_mb = torch.cat((self.lbl_mb[self.mb_dequeue_point[0]:], s_lbl, lt_plbl))
            self.mb_dequeue_point.remove(self.mb_dequeue_point[0])
            self.mb_dequeue_point.append(len(embs_s) + len(embs_lt))

    def contrastive_loss(self, emb_s, lbl_s, emb_lt, lbl_lt):
        emb_s = nn.functional.normalize(emb_s, dim=1)
        sim_mat_s = emb_s @ self.emb_mb.T
        sim_mat_s = torch.exp(sim_mat_s / self.T)
        sim_mat_s_aug = emb_s @ self.emb_mb_aug.T
        sim_mat_s_aug = torch.exp(sim_mat_s_aug / self.T)

        pos_mask_s = ((lbl_s.expand(self.emb_mb.shape[0], emb_s.shape[0]).T).eq(
            self.lbl_mb.expand(emb_s.shape[0], self.emb_mb.shape[0]))).to(torch.float32).cuda()
        pos_mask_s_aug = ((lbl_s.expand(self.emb_mb_aug.shape[0], emb_s.shape[0]).T).eq(
            self.lbl_mb.expand(emb_s.shape[0], self.emb_mb_aug.shape[0]))).to(torch.float32).cuda()
        neg_mask_s = torch.ones((emb_s.shape[0], self.emb_mb.shape[0]), dtype=torch.float32).cuda() - pos_mask_s
        neg_mask_s_aug = torch.ones((emb_s.shape[0], self.emb_mb_aug.shape[0]),
                                    dtype=torch.float32).cuda() - pos_mask_s_aug

        pos_s = (sim_mat_s * pos_mask_s).sum(dim=1) + 0.00001
        neg_s = (sim_mat_s * neg_mask_s).sum(dim=1) + (sim_mat_s_aug * neg_mask_s_aug).sum(dim=1) + 0.00001
        cont_loss_s = (-1.0 * (torch.log(pos_s / (pos_s + neg_s)))).mean()

        emb_lt = nn.functional.normalize(emb_lt, dim=1)
        sim_mat_lt = emb_lt @ self.emb_mb.T
        sim_mat_lt = torch.exp(sim_mat_lt / self.T)
        sim_mat_lt_aug = emb_lt @ self.emb_mb_aug.T
        sim_mat_lt_aug = torch.exp(sim_mat_lt_aug / self.T)

        pos_mask_lt = ((lbl_lt.expand(self.emb_mb.shape[0], emb_lt.shape[0]).T).eq(
            self.lbl_mb.expand(emb_lt.shape[0], self.emb_mb.shape[0]))).to(torch.float32).cuda()
        pos_mask_lt_aug = ((lbl_lt.expand(self.emb_mb_aug.shape[0], emb_lt.shape[0]).T).eq(
            self.lbl_mb.expand(emb_lt.shape[0], self.emb_mb_aug.shape[0]))).to(torch.float32).cuda()
        neg_mask_lt = torch.ones((emb_lt.shape[0], self.emb_mb.shape[0]), dtype=torch.float32).cuda() - pos_mask_lt
        neg_mask_lt_aug = torch.ones((emb_lt.shape[0], self.emb_mb_aug.shape[0]),
                                     dtype=torch.float32).cuda() - pos_mask_lt_aug

        pos_lt = (sim_mat_lt * pos_mask_lt).sum(dim=1) + 0.000001
        neg_lt = (sim_mat_lt * neg_mask_lt).sum(dim=1) + (sim_mat_lt_aug * neg_mask_lt_aug).sum(dim=1) + 0.000001
        cont_loss_lt = (-1.0 * (torch.log(pos_lt / (pos_lt + neg_lt)))).mean()

        return cont_loss_s, cont_loss_lt

    @torch.no_grad()
    def update_mb_reg(self, t_img, t_img_s):
        embs_t_reg, _ = self.momentum_model(t_img, getemb=True)
        embs_t_reg = nn.functional.normalize(embs_t_reg, dim=1)
        embs_t_reg_aug, _ = self.momentum_model(t_img_s, getemb=True)
        embs_t_reg_aug = nn.functional.normalize(embs_t_reg_aug, dim=1)

        if self.mb_reg_init_sum == 0:
            self.emb_mb_reg = embs_t_reg
            self.emb_mb_reg_aug = embs_t_reg_aug
            self.mb_reg_dequeue_point.append(len(embs_t_reg))
            self.mb_reg_init_sum = self.mb_reg_init_sum + 1
        elif self.mb_reg_init_sum > 0 and self.mb_reg_init_sum < self.mb_reg_init_max_iter:
            self.emb_mb_reg = torch.cat((self.emb_mb_reg, embs_t_reg))
            self.emb_mb_reg_aug = torch.cat((self.emb_mb_reg_aug, embs_t_reg_aug))
            self.mb_reg_dequeue_point.append(len(embs_t_reg))
            self.mb_reg_init_sum = self.mb_reg_init_sum + 1
        else:
            self.emb_mb_reg = torch.cat((self.emb_mb_reg[self.mb_reg_dequeue_point[0]:], embs_t_reg))
            self.emb_mb_reg_aug = torch.cat((self.emb_mb_reg_aug[self.mb_reg_dequeue_point[0]:], embs_t_reg_aug))
            self.mb_reg_dequeue_point.remove(self.mb_reg_dequeue_point[0])
            self.mb_reg_dequeue_point.append(len(embs_t_reg))

    def train_one_epoch(self, epoch):
        self.model.train()
        print(len(self.labeled_target_loader.dataset))
        print(self.labeled_target_loader.batch_size)
        src_iter = ForeverDataIterator(self.source_train_loader)
        src_cb_iter = ForeverDataIterator(self.source_cb_loader)
        trg_iter = iter(self.target_train_loader)
        ltrg_iter = ForeverDataIterator(self.labeled_target_loader)
        lt_cb_iter = ForeverDataIterator(self.labeled_target_cb_loader)
        for step in range(len(self.target_train_loader)):
            s_data = next(src_iter)
            s_img = s_data[0].cuda()
            s_img_s = s_data[3].cuda()
            s_gtlbl = s_data[1].cuda()

            t_data = next(trg_iter)
            t_img = t_data[0].cuda()
            t_img_s = t_data[3].cuda()

            lt_data = next(ltrg_iter)
            lt_img = lt_data[0].cuda()
            lt_img_s = lt_data[3].cuda()
            lt_gtlbl = lt_data[1].cuda()

            s_cb_data = next(src_cb_iter)

            lt_cb_data = next(lt_cb_iter)

            emb_s, output_s = self.model(s_img, getemb=True)
            loss_s = F.cross_entropy(output_s, s_gtlbl)

            embs_lt, output_lt = self.model(lt_img, getemb=True)
            loss_lt = F.cross_entropy(output_lt, lt_gtlbl)

            self.update_mb(epoch, step, s_cb_data, lt_cb_data)

            embs_s_s, _ = self.model(s_img_s, getemb=True)
            embs_lt_s, _ = self.model(lt_img_s, getemb=True)
            cont_loss_s, cont_loss_lt = self.contrastive_loss(embs_s_s, s_gtlbl, embs_lt_s, lt_gtlbl)

            if epoch > 0 or step > 0:
                with torch.no_grad():
                    embs_t_reg, output_t_reg = self.momentum_model(t_img, getemb=True)
                    embs_t_reg = nn.functional.normalize(embs_t_reg, dim=1)
                embs_t_reg_s, output_t_reg_s = self.model(t_img_s, getemb=True)
                embs_t_reg_s = nn.functional.normalize(embs_t_reg_s, dim=1)

                sim_mat_reg = embs_t_reg_s @ embs_t_reg.T
                pos_mask_reg = torch.eye(embs_t_reg_s.shape[0]).cuda()
                pos_reg = (sim_mat_reg * pos_mask_reg).sum(dim=1) + 0.00001

                sim_mat_mb_reg = embs_t_reg_s @ self.emb_mb_reg.T
                neg_mb_reg = sim_mat_mb_reg.sum(dim=1) + 0.00001

                sim_mat_mb_reg_s = embs_t_reg_s @ self.emb_mb_reg_aug.T
                neg_mb_reg_s = sim_mat_mb_reg_s.sum(dim=1) + 0.00001

                cont_loss_reg = (-1.0 * (torch.log(pos_reg / (pos_reg + neg_mb_reg + neg_mb_reg_s)))).mean()
                self.update_mb_reg(t_img, t_img_s)
            else:
                cont_loss_reg = torch.tensor(0.0).cuda()
                self.update_mb_reg(t_img, t_img_s)

            if epoch in [1, 3, 5, 7] or epoch > self.sampling_epochs[-1]:
                # VisDA and MiniDomainNet
                # if step >= int(len(self.target_train_loader) / 2) or epoch > self.sampling_epochs[-1]:
                embs_t, output_t = self.model(t_img, getemb=True)
                embs_l = torch.cat((emb_s, embs_lt))
                output_l = torch.cat(
                    (torch.softmax(output_s.detach(), dim=1), torch.softmax(output_lt.detach(), dim=1)))
                if epoch in [1, 3, 5, 7, 9] and step == 0:
                    # VisDA and MiniDomainNet
                    # if epoch in [0, 1, 2, 3, 4] and step == int(len(self.target_train_loader) / 2):
                    self.prototype_all_ut = (torch.softmax(output_t.detach(), dim=1).T @ embs_t) / (
                            1e-8 + torch.softmax(output_t.detach(), dim=1).sum(axis=0)[:, None])
                    self.prototype_all_l = ((torch.softmax(output_l, dim=1)).T @ embs_l) / (
                            1e-8 + (torch.softmax(output_l, dim=1)).sum(axis=0)[:, None])
                else:
                    cur_prototype_all_ut = (torch.softmax(output_t.detach(), dim=1).T @ embs_t) / (
                            1e-8 + torch.softmax(output_t.detach(), dim=1).sum(axis=0)[:, None])
                    self.prototype_all_ut = self.args.ema * self.prototype_all_ut.detach() + (
                            1 - self.args.ema) * cur_prototype_all_ut
                    cur_prototype_all_l = ((torch.softmax(output_l, dim=1)).T @ embs_l) / (
                            1e-8 + (torch.softmax(output_l, dim=1)).sum(axis=0)[:, None])
                    self.prototype_all_l = self.args.ema * self.prototype_all_l.detach() + (
                            1 - self.args.ema) * cur_prototype_all_l

                sim_mat = nn.functional.normalize(self.prototype_all_ut, dim=1) @ nn.functional.normalize(
                    self.prototype_all_l, dim=1).T
                sim_mat = torch.exp(sim_mat / self.T)

                pos_mask = torch.eye(self.args.class_num).cuda()
                neg_mask = torch.ones(self.args.class_num, self.args.class_num).cuda() - pos_mask

                pos = (sim_mat * pos_mask).sum(dim=1) + 0.00001
                neg = (sim_mat * neg_mask).sum(dim=1) + 0.00001
                cont_loss_t = (-1.0 * (torch.log(pos / (pos + neg)))).mean()

                loss = loss_s + loss_lt + self.args.cont_loss_weight1 * (
                        cont_loss_s + cont_loss_lt + cont_loss_reg) + self.args.cont_loss_weight2 * cont_loss_t
            else:
                loss = loss_s + loss_lt + self.args.cont_loss_weight1 * (cont_loss_s + cont_loss_lt + cont_loss_reg)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.scheduler.step()
            if step % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, step * len(s_gtlbl),
                                                                              len(self.target_train_loader.dataset),
                                                                              100. * step / len(
                                                                                  self.target_train_loader),
                                                                              loss.item()))

    def probability_difference(self, original_pro_output, reconfiguration_pro_output):
        original_pred_cls = original_pro_output.argmax()
        reconfiguration_pred_cls_all = reconfiguration_pro_output.argmax(dim=1)
        if original_pred_cls != reconfiguration_pred_cls_all:
            is_change = True
        else:
            is_change = False
        return (-(original_pro_output + reconfiguration_pro_output) * torch.log(
            original_pro_output * reconfiguration_pro_output)).sum(), is_change, original_pro_output.max().item(), original_pred_cls

    def get_labeled_target_dataset_per_cls_sum(self):
        per_cls_sum = np.zeros(self.args.class_num)
        for i in range(len(self.labeled_target_ds)):
            _, target = self.labeled_target_ds.samples[i]
            per_cls_sum[target] = per_cls_sum[target] + 1
        return per_cls_sum

    def sample_selection(self, epoch):
        self.pseudo_labeled_target_ds = ImageList(transform=train_transform, strong_transform=strong_transform,
                                                  empty=True)
        if epoch > 0:
            pseudo_labeled_target_cb_ds = deepcopy(self.labeled_target_ds)
        else:
            pseudo_labeled_target_cb_ds = ImageList(transform=train_transform, strong_transform=strong_transform,
                                                    empty=True)

        self.source_stand_dataset = deepcopy(self.source_train_ds)
        self.source_stand_dataset.transform = test_transform
        source_stand_loader = DataLoader(self.source_stand_dataset, batch_size=self.args.batch_size, shuffle=False,
                                         drop_last=False, num_workers=2)

        self.target_stand_dataset = deepcopy(self.target_train_ds)
        self.target_stand_dataset.transform = test_transform
        target_stand_loader = DataLoader(self.target_stand_dataset, batch_size=self.args.batch_size, shuffle=False,
                                         drop_last=False, num_workers=2)

        model_pro_output_all = torch.zeros((len(self.target_stand_dataset), self.args.class_num))
        feat_all = torch.zeros((len(self.target_stand_dataset), self.inc))
        prototype_source = torch.zeros((self.args.class_num, self.inc))
        each_class_sum = torch.zeros((self.args.class_num, 1))
        class_difficulty = torch.ones(self.args.class_num)
        target_stand_loader_iter = iter(target_stand_loader)
        source_stand_loader_iter = iter(source_stand_loader)

        self.model = self.model.cuda()
        self.model.eval()
        stat = list()
        with torch.no_grad():
            for batch_idx in range(len(source_stand_loader)):
                data_s = next(source_stand_loader_iter)
                im_data_s = data_s[0].cuda()
                label_s = data_s[1]
                feat, _ = self.model(im_data_s, getfeat=True)
                feat = feat.cpu()
                for i in range(len(label_s)):
                    prototype_source[int(label_s[i])] += feat[i]
                    each_class_sum[int(label_s[i])] += 1
            prototype_source = prototype_source / each_class_sum

            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                im_data_t = data_t[0].cuda()
                feat, output1 = self.model(im_data_t, getfeat=True)
                softmax_out = torch.softmax(output1, dim=1)
                if batch_idx == (len(target_stand_loader) - 1):
                    feat_all[batch_idx * self.args.batch_size:] = feat.cpu()
                    model_pro_output_all[batch_idx * self.args.batch_size:] = softmax_out.cpu()
                else:
                    feat_all[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] = feat.cpu()
                    model_pro_output_all[
                    batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] = softmax_out.cpu()

            for i in range(self.args.class_num):
                class_difficulty[i] = (((model_pro_output_all.max(dim=1)[1] == i) * model_pro_output_all[:,
                                                                                    i]).sum()) / (
                                          (model_pro_output_all.max(dim=1)[1] == i).sum())

            target_stand_loader_iter = iter(target_stand_loader)
            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                labels = data_t[1]
                paths = data_t[2]
                for j in range(len(labels)):
                    current_index = batch_idx * self.args.batch_size + j
                    current_original_pro_output = model_pro_output_all[current_index]

                    current_reconfiguration_feat_source = current_original_pro_output @ prototype_source
                    current_reconfiguration_feat_source = current_reconfiguration_feat_source.cuda()
                    current_reconfiguration_feat_source = current_reconfiguration_feat_source.unsqueeze(0)
                    _, current_reconfiguration_pro_output_source = self.model(current_reconfiguration_feat_source,
                                                                              getemb=True, justclf=True)
                    current_reconfiguration_pro_output_source = torch.softmax(current_reconfiguration_pro_output_source,
                                                                              dim=1)
                    current_reconfiguration_pro_output_source = current_reconfiguration_pro_output_source.cpu()
                    score1, is_change_source, confidence, pl = self.probability_difference(current_original_pro_output,
                                                                                           current_reconfiguration_pro_output_source)

                    current_path = paths[j]
                    current_target = labels[j]

                    stat.append(
                        [current_path, current_target, current_index, score1, is_change_source, confidence, pl])

            stat = np.array(stat, dtype=object)
            score1_min = stat[:, 3].min().item()
            score1_max = stat[:, 3].max().item()
            stat[:, 3] = (stat[:, 3] - score1_min) / (score1_max - score1_min)
            stat = sorted(stat, key=lambda x: x[3], reverse=True)
            pl_stat = stat[int(len(stat) / 2):]
            stat = stat[:int(len(stat) / 2)]

            stat = np.array(stat, dtype=object)
            index = list()
            active_sum = 0
            candicate_ds_index = list()
            for i in range(len(stat)):
                if active_sum == self.num_anno:
                    break
                if np.random.random() >= class_difficulty[stat[i, 6, ...].item()]:
                    candicate_ds_index.append(stat[i, 2, ...])
                    index.append(i)
                    active_sum = active_sum + 1
            active_samples = stat[index, 0:3, ...]

            print("len_active_sample = {}".format(len(index)))
            active_samples = np.array(active_samples, dtype=object)
            candicate_ds_index = np.array(candicate_ds_index, dtype=object)
            self.labeled_target_ds.add_item(active_samples[:, 0:2, ...])
            self.target_train_ds.remove_item(candicate_ds_index.astype('int64'))

            per_cls_sum = self.get_labeled_target_dataset_per_cls_sum()
            max_cls_num = per_cls_sum.max()
            pl_stat = sorted(pl_stat, key=lambda x: x[3], reverse=False)
            pl_stat = np.array(pl_stat, dtype=object)
            for i in range(len(pl_stat)):
                if pl_stat[i, 4, ...] == False and pl_stat[i, 5, ...] >= 0.95 and per_cls_sum[
                    int(pl_stat[i, 6, ...])] < max_cls_num:
                    self.pseudo_labeled_target_ds.add_item(pl_stat[i, (0, 6), ...][None])
                    pseudo_labeled_target_cb_ds.add_item(pl_stat[i, (0, 6), ...][None])
                    per_cls_sum[int(pl_stat[i, 6, ...])] = per_cls_sum[int(pl_stat[i, 6, ...])] + 1

        self.target_train_loader = DataLoader(self.target_train_ds, batch_size=self.args.batch_size, shuffle=True,
                                              drop_last=True, num_workers=2)
        self.pseudo_labeled_target_ds.add_item(self.labeled_target_ds.samples)
        self.labeled_target_loader = DataLoader(self.pseudo_labeled_target_ds, batch_size=min(self.args.batch_size,
                                                                                              len(self.pseudo_labeled_target_ds)),
                                                shuffle=True, drop_last=True, num_workers=2)
        self.lt_cb_sampler = sampler.get_sampler({
            'dataset': pseudo_labeled_target_cb_ds,
            'num_categories': self.args.class_num,
            'n_samples': self.args.n_samples,
        })
        self.labeled_target_cb_loader = DataLoader(pseudo_labeled_target_cb_ds, batch_sampler=self.lt_cb_sampler,
                                                   shuffle=False, num_workers=2)

    def update_plt(self, epoch):
        self.pseudo_labeled_target_ds = ImageList(transform=train_transform, strong_transform=strong_transform,
                                                  empty=True)

        self.source_stand_dataset = deepcopy(self.source_train_ds)
        self.source_stand_dataset.transform = test_transform
        source_stand_loader = DataLoader(self.source_stand_dataset, batch_size=self.args.batch_size, shuffle=False,
                                         drop_last=False, num_workers=2)

        self.target_stand_dataset = deepcopy(self.target_train_ds)
        self.target_stand_dataset.transform = test_transform
        target_stand_loader = DataLoader(self.target_stand_dataset, batch_size=self.args.batch_size, shuffle=False,
                                         drop_last=False, num_workers=2)

        model_pro_output_all = torch.zeros((len(self.target_stand_dataset), self.args.class_num))
        feat_all = torch.zeros((len(self.target_stand_dataset), self.inc))
        prototype_source = torch.zeros((self.args.class_num, self.inc))
        each_class_sum = torch.zeros((self.args.class_num, 1))
        target_stand_loader_iter = iter(target_stand_loader)
        source_stand_loader_iter = iter(source_stand_loader)

        self.model = self.model.cuda()
        self.model.eval()
        stat = list()
        with torch.no_grad():
            for batch_idx in range(len(source_stand_loader)):
                data_s = next(source_stand_loader_iter)
                im_data_s = data_s[0].cuda()
                label_s = data_s[1]
                feat, _ = self.model(im_data_s, getfeat=True)
                feat = feat.cpu()
                for i in range(len(label_s)):
                    prototype_source[int(label_s[i])] += feat[i]
                    each_class_sum[int(label_s[i])] += 1
            prototype_source = prototype_source / each_class_sum

            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                im_data_t = data_t[0].cuda()
                feat, output1 = self.model(im_data_t, getfeat=True)
                softmax_out = torch.softmax(output1, dim=1)
                if batch_idx == (len(target_stand_loader) - 1):
                    feat_all[batch_idx * self.args.batch_size:] = feat.cpu()
                    model_pro_output_all[batch_idx * self.args.batch_size:] = softmax_out.cpu()
                else:
                    feat_all[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] = feat.cpu()
                    model_pro_output_all[
                    batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] = softmax_out.cpu()

            target_stand_loader_iter = iter(target_stand_loader)
            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                labels = data_t[1]
                paths = data_t[2]
                for j in range(len(labels)):
                    current_index = batch_idx * self.args.batch_size + j
                    current_original_pro_output = model_pro_output_all[current_index]

                    current_reconfiguration_feat_source = current_original_pro_output @ prototype_source
                    current_reconfiguration_feat_source = current_reconfiguration_feat_source.cuda()
                    current_reconfiguration_feat_source = current_reconfiguration_feat_source.unsqueeze(0)
                    _, current_reconfiguration_pro_output_source = self.model(current_reconfiguration_feat_source,
                                                                              getemb=True, justclf=True)
                    current_reconfiguration_pro_output_source = torch.softmax(current_reconfiguration_pro_output_source,
                                                                              dim=1)
                    current_reconfiguration_pro_output_source = current_reconfiguration_pro_output_source.cpu()
                    score1, is_change_source, confidence, pl = self.probability_difference(current_original_pro_output,
                                                                                           current_reconfiguration_pro_output_source)

                    current_path = paths[j]
                    current_target = labels[j]

                    stat.append(
                        [current_path, current_target, current_index, score1, is_change_source, confidence, pl])

            stat = np.array(stat, dtype=object)
            score1_min = stat[:, 3].min().item()
            score1_max = stat[:, 3].max().item()
            stat[:, 3] = (stat[:, 3] - score1_min) / (score1_max - score1_min)
            stat = sorted(stat, key=lambda x: x[3], reverse=True)
            pl_stat = stat[int(len(stat) / 2):]

            per_cls_sum = self.get_labeled_target_dataset_per_cls_sum()
            max_cls_num = per_cls_sum.max()
            pl_stat = sorted(pl_stat, key=lambda x: x[3], reverse=False)
            pl_stat = np.array(pl_stat, dtype=object)
            for i in range(len(pl_stat)):
                if pl_stat[i, 4, ...] == False and pl_stat[i, 5, ...] >= 0.95 and per_cls_sum[
                    int(pl_stat[i, 6, ...])] < max_cls_num:
                    self.pseudo_labeled_target_ds.add_item(pl_stat[i, (0, 6), ...][None])
                    per_cls_sum[int(pl_stat[i, 6, ...])] = per_cls_sum[int(pl_stat[i, 6, ...])] + 1

        self.target_train_loader = DataLoader(self.target_train_ds, batch_size=self.args.batch_size, shuffle=True,
                                              drop_last=True, num_workers=2)
        self.pseudo_labeled_target_ds.add_item(self.labeled_target_ds.samples)
        self.labeled_target_loader = DataLoader(self.pseudo_labeled_target_ds, batch_size=min(self.args.batch_size,
                                                                                              len(self.pseudo_labeled_target_ds)),
                                                shuffle=True, drop_last=True, num_workers=2)
        self.lt_cb_sampler = sampler.get_sampler({
            'dataset': self.pseudo_labeled_target_ds,
            'num_categories': self.args.class_num,
            'n_samples': self.args.n_samples,
        })
        self.labeled_target_cb_loader = DataLoader(self.pseudo_labeled_target_ds, batch_sampler=self.lt_cb_sampler,
                                                   shuffle=False, num_workers=2)

    def train(self):
        self.optim.zero_grad()
        self.test()
        for epoch in range(self.args.epochs):
            if epoch in self.sampling_epochs:
                self.sample_selection(epoch)
            else:
                self.update_plt(epoch)
            self.train_one_epoch(epoch)
            self.test()

    def pre_train(self):
        self.optim.zero_grad()
        self.test()
        for epoch in range(self.args.epochs):
            self.pre_train_one_epoch(epoch)
            self.test()
        torch.save(self.model, osp.join(self.args.save_model_path, "pre_train_model.pth.tar"))
