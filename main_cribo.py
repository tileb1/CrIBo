# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
import einops

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange, repeat
from torchvision import models as torchvision_models

from cribo_utils.data_transforms import *
from cribo_utils.datasets import get_dataloader, get_dataset
from cribo_utils.hpc import pin_workers_iterator
from source.models import vision_transformer as vits
from source.models import swin_transformer as swins
from source.models.vision_transformer import DINOHead
from source.utils import utils

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

from cribo_utils.parser import get_args_parser


def train_cribo(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationCrIBo(
        global_crops_scale=args.global_crops_scale,
        same_teacher_augmentations=args.same_teacher_augmentations
    )

    dataset = get_dataset(args, transform, val_or_train='train')
    data_loader = get_dataloader(args, dataset)

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.drop_path_rate,
                                           is_teacher=False)
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in swins.__dict__.keys():
        student = swins.__dict__[args.arch]()
        teacher = swins.__dict__[args.arch]()
        embed_dim = student.num_features
        args.patch_size = 32  # Cheap trick
    elif 'resnet' in args.arch:
        from source.models import resnet
        args.patch_size = 32  # Cheap trick
        student, embed_dim = resnet.__dict__[args.arch](zero_init_residual=True)
        teacher, _ = resnet.__dict__[args.arch](zero_init_residual=True)
    else:
        print(f"Unknow architecture: {args.arch}")
        raise NotImplementedError

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapperDINO(
        student,
        DINOHead(embed_dim, args.out_dim, args.out_dim_c, use_bn=args.use_bn_in_head,
                 norm_last_layer=args.norm_last_layer),
        args.which_features,
    )
    teacher = utils.MultiCropWrapperDINO(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.out_dim_c, args.use_bn_in_head),
        args.which_features,
    )
    clustering = Clustering(
        args,
        n_tokens=args.n_tokens,
        n_heads=1 if args.which_features == "last" else teacher.backbone.num_heads,
        sinkhorn_lambda=args.sinkhorn_lambda,
        sinkhorn_iterations=args.sinkhorn_iterations,
        student_temp=args.student_temp,
        pos_alpha=args.pos_alpha,
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False,
                                                  broadcast_buffers=False)

    # teacher and student start with the same weights
    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    print('Teacher loaded with msg: {}'.format(msg))
    print('Teacher loaded with msg: {}'.format(msg))

    # there is no backpropagation through the teacher, so no need for gradients
    for n, p in teacher.named_parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    cls_queue_cpu = torch.rand(2, args.queue_size, embed_dim, dtype=torch.float)
    cpv_queue_cpu = args.n_tokens * torch.ones(args.queue_size, dtype=torch.int)
    centroids_queue_cpu = torch.rand(2, args.n_tokens * args.queue_size, embed_dim, dtype=torch.float)

    # ============ preparing losses ... ============
    dino_loss = CrIBoLoss(
        out_dim=args.out_dim,
        out_dim_c=args.out_dim_c,
        ncrops=args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp=args.warmup_teacher_temp,
        warmup_teacher_temp_c=args.warmup_teacher_temp_c,
        teacher_temp=args.teacher_temp,
        teacher_temp_c=args.teacher_temp_c,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp_c=args.student_temp_c,
        args=args
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise NotImplementedError
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============n
    to_restore = {
        "epoch": 0,
        'centroids_queue_cpu': centroids_queue_cpu,
        'cls_queue_cpu': cls_queue_cpu,
        'cpv_queue_cpu': cpv_queue_cpu,
    }
    default_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if not os.path.isfile(default_checkpoint_path) and os.path.isfile(args.start_checkpoint_path):
        utils.master_copy_from_to(args.start_checkpoint_path, default_checkpoint_path)

    torch.distributed.barrier()

    try:
        utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    except:
        # If checkpoint is corrupted, used backedup checkpoint
        utils.restart_from_checkpoint(
            default_checkpoint_path + '.backup',
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    start_epoch = to_restore["epoch"]
    centroids_queue_cpu = to_restore["centroids_queue_cpu"]
    cls_queue_cpu = to_restore["cls_queue_cpu"]
    cpv_queue_cpu = to_restore["cpv_queue_cpu"]

    centroids_queue = centroids_queue_cpu.to(args.gpu)
    cls_queue = cls_queue_cpu.to(args.gpu)
    cpv_queue = cpv_queue_cpu.to(args.gpu)

    start_time = time.time()
    print("Starting CrIBo training !")
    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CrIBo ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer,
                                      lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, clustering, args, centroids_queue, cls_queue, cpv_queue)

        centroids_queue_cpu = centroids_queue.cpu()
        cls_queue_cpu = cls_queue.cpu()
        cpv_queue_cpu = cpv_queue.cpu()

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'centroids_queue_cpu': centroids_queue_cpu,
            'cls_queue_cpu': cls_queue_cpu,
            'cpv_queue_cpu': cpv_queue_cpu,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                     'epoch_time': time.time() - start_epoch_time}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.barrier()
    dist.destroy_process_group()


def all_gather(q, ws, return_lst=False, return_cat=True):
    """
    Gathers tensor arrays of different lengths across multiple gpus

    Parameters
    ----------
        q : tensor array
        ws : world size
        device : current gpu device

    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.shape[0], device=q.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size - q.shape[0]
    if size_diff > 0:
        padding = torch.zeros(size_diff, q.shape[1], device=q.device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    output_l = []
    for q, size in zip(all_qs_padded, all_sizes):
        output_l.append(q[:size])

    returned = []
    if return_cat:
        out = torch.cat(output_l)
        returned.append(out)
    if return_lst:
        returned.append(output_l)
    returned = returned[0] if len(returned) == 1 else tuple(returned)
    return returned


@torch.no_grad()
def get_nn_centroids(teacher_cls, teacher_centroids, centroids_per_view, centroids_queue, cls_queue, cpv_queue, args):
    # Unpack and normalize
    teacher_centroids_v1_orig, teacher_centroids_v2_orig = teacher_centroids
    teacher_cls_v1, teacher_cls_v2 = teacher_cls.chunk(2)
    if args.normalized_matching:
        teacher_cls_v1 = F.normalize(teacher_cls_v1, p=2, dim=-1)
        teacher_cls_v2 = F.normalize(teacher_cls_v2, p=2, dim=-1)

    # Create the mask
    centroids_stop_queue = torch.cumsum(cpv_queue, dim=0)
    centroids_start_queue = torch.zeros_like(centroids_stop_queue)
    centroids_start_queue[1:] = centroids_stop_queue[:-1].clone()

    # Infer the number of valid centroids
    n_valid_queue_centroids = cpv_queue.sum()

    # Normalize (the [CLS] queue stores normalized embeddings)
    teacher_centroids_v1 = F.normalize(teacher_centroids_v1_orig, p=2, dim=-1)
    teacher_centroids_v2 = F.normalize(teacher_centroids_v2_orig, p=2, dim=-1)
    if args.normalized_matching:
        centroids_queue_normalized = F.normalize(centroids_queue[:, :n_valid_queue_centroids], p=2, dim=-1)
    else:
        centroids_queue_normalized = centroids_queue[:, :n_valid_queue_centroids].clone()

    # Outgoing NN search
    centroids_similarity = teacher_centroids_v1 @ centroids_queue_normalized[0].T
    centroids_val_knn_local1, centroids_indices_knn_local1 = centroids_similarity.topk(dim=-1, k=1)
    centroids_indices_knn_local1 = centroids_indices_knn_local1[:, 0]

    # Updown
    centroids_similarity2 = teacher_centroids_v2 @ centroids_queue_normalized[1].T
    _, centroids_indices_knn_local2 = centroids_similarity2.topk(dim=-1, k=1)
    centroids_indices_knn_local2 = centroids_indices_knn_local2[:, 0]

    # Cycle
    centroids_retrieved_nn = centroids_queue_normalized[1][centroids_indices_knn_local1]
    centroids_similarity3 = centroids_retrieved_nn @ teacher_centroids_v2.T
    _, centroids_indices_knn_local3 = centroids_similarity3.topk(dim=-1, k=1)
    centroids_indices_knn_local3 = centroids_indices_knn_local3[:, 0]

    # Cycle 2
    centroids_retrieved_nn2 = centroids_queue_normalized[0][centroids_indices_knn_local2]
    centroids_similarity4 = centroids_retrieved_nn2 @ teacher_centroids_v1.T
    _, centroids_indices_knn_local4 = centroids_similarity4.topk(dim=-1, k=1)
    centroids_indices_knn_local4 = centroids_indices_knn_local4[:, 0]

    # Infer the condition
    centroids_cycle_condition = centroids_indices_knn_local3 == torch.IntTensor(
        list(range(len(centroids_indices_knn_local3)))).to(args.gpu)

    # Infer the condition 2
    centroids_cycle_condition2 = centroids_indices_knn_local4 == torch.IntTensor(
        list(range(len(centroids_indices_knn_local4)))).to(args.gpu)

    bootstrap_centroid_mask = centroids_cycle_condition
    bootstrap_centroid_mask2 = centroids_cycle_condition2

    # Retrieve non normalized NN (view 1 and 2)
    nn = centroids_queue[:, centroids_indices_knn_local1]
    nn2 = centroids_queue[:, centroids_indices_knn_local2]

    # Collect the number of centroids per view from all gpus
    centroids_per_view_full = [torch.zeros_like(centroids_per_view) for _ in range(args.world_size)]
    dist.all_gather(centroids_per_view_full, centroids_per_view)
    centroids_per_view_full = torch.cat(centroids_per_view_full)

    # Collect the [CLS]
    cls_v1_full = [torch.zeros_like(teacher_cls_v1) for _ in range(args.world_size)]
    cls_v2_full = [torch.zeros_like(teacher_cls_v2) for _ in range(args.world_size)]
    dist.all_gather(cls_v1_full, teacher_cls_v1)
    dist.all_gather(cls_v2_full, teacher_cls_v2)
    cls_v1_full = torch.cat(cls_v1_full)
    cls_v2_full = torch.cat(cls_v2_full)

    # Update queues
    pool_1, lst_1 = all_gather(teacher_centroids_v1_orig, args.world_size, return_lst=True)
    pool_2, lst_2 = all_gather(teacher_centroids_v2_orig, args.world_size, return_lst=True)

    if len(cls_v1_full) >= args.queue_size:
        cls_v1_full = cls_v1_full[:args.queue_size]
        cls_v2_full = cls_v2_full[:args.queue_size]
        centroids_per_view_full = centroids_per_view_full[:args.queue_size]
        last_centroid = centroids_per_view_full.sum()
        pool_1 = pool_1[:last_centroid].clone()
        pool_2 = pool_2[:last_centroid].clone()

    n_old_cls = len(cls_queue[0]) - len(cls_v1_full)
    n_new_centroids = centroids_per_view_full.sum()
    n_old_centroids = cpv_queue[:n_old_cls].sum()

    cls_queue[0] = torch.cat([cls_v1_full, cls_queue[0, :-len(cls_v1_full)]])
    cls_queue[1] = torch.cat([cls_v2_full, cls_queue[1, :-len(cls_v2_full)]])
    centroids_queue[0, : n_new_centroids + n_old_centroids] = torch.cat((pool_1, centroids_queue[0, :n_old_centroids]))
    centroids_queue[1, : n_new_centroids + n_old_centroids] = torch.cat((pool_2, centroids_queue[1, :n_old_centroids]))

    cpv_queue[len(cls_v1_full):] = cpv_queue[:-len(cls_v1_full)].clone()
    cpv_queue[:len(cls_v1_full)] = centroids_per_view_full

    return nn, bootstrap_centroid_mask, centroids_indices_knn_local1, nn2, centroids_indices_knn_local2, bootstrap_centroid_mask2


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, lr_schedule, wd_schedule,
                    momentum_schedule, epoch, fp16_scaler, clustering, args, centroids_queue, cls_queue, cpv_queue):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    the_iterator = iter(data_loader)
    pin_workers_iterator(the_iterator, args)

    for it, to_unpack in enumerate(metric_logger.log_every(the_iterator, 20, header)):
        if len(to_unpack):
            images, indices, crop_pos = to_unpack
        else:
            raise NotImplementedError

        crop_pos = [p.cuda(non_blocking=True) for p in crop_pos]
        crop_pos = rearrange(torch.stack(crop_pos[:2]), 'm b d (r i) (c j) -> m b d (r c) (i j)', i=args.patch_size,
                             j=args.patch_size).mean(dim=-1)
        images = [im.cuda(non_blocking=True) for im in images]

        it_ = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it_]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            with torch.no_grad():
                # Get the teacher tokens
                teacher_output, teacher_features, teacher_last_tokens, teacher_tokens = teacher(images)

                # Compute the teacher's assignments and centroids
                teacher_centroids, valid_centroids, assignments, region_count, centroids_per_view = clustering. \
                    compute_teacher_centroids(teacher_last_tokens, teacher_tokens, crop_pos, epoch,
                                              use_hard_assignment=True)

            # Get the student tokens
            student_output, student_features, student_last_tokens, _ = student(images)

            # Compute the student's assignments and centroids
            student_centroids = clustering.compute_student_centroids(assignments, student_last_tokens, valid_centroids)

            nn, bootstrap_centroid_mask, centroids_indices_knn_local1, nn2, centroids_indices_knn_local2, bootstrap_centroid_mask2 = get_nn_centroids(
                teacher_features, teacher_centroids, centroids_per_view, centroids_queue, cls_queue, cpv_queue, args)

            # Project the centroids
            with torch.no_grad():
                teacher_centroids = teacher(torch.cat(teacher_centroids), head_only=True)
                teacher_nn_centroids = teacher(nn, head_only=True)
                teacher_nn2_centroids = teacher(nn2, head_only=True)
            student_centroids = student(torch.cat(student_centroids), head_only=True)

            # Get the [CLS] loss
            d_loss, s_loss, nn_loss = dino_loss(student_output, teacher_output, epoch, student_centroids,
                                                teacher_centroids, bootstrap_centroid_mask, teacher_nn_centroids,
                                                centroids_indices_knn_local1, teacher_nn2_centroids,
                                                centroids_indices_knn_local2, bootstrap_centroid_mask2)

            # Combine the losses
            loss = d_loss + s_loss + nn_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # student update
        param_norms = None
        if fp16_scaler is None:
            # Back-propagate the loss
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Back-propagate the loss
            fp16_scaler.scale(loss).backward()

            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it_]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(d_loss=d_loss.item())
        metric_logger.update(s_loss=s_loss.item())
        metric_logger.update(nn_loss=nn_loss.item())
        metric_logger.update(r_cnt=region_count)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(bootstrap_centroid_ratio=bootstrap_centroid_mask.sum() / bootstrap_centroid_mask.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class CrIBoLoss(nn.Module):
    def __init__(self, out_dim, out_dim_c, ncrops, warmup_teacher_temp, warmup_teacher_temp_c, teacher_temp,
                 teacher_temp_c,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, student_temp_c=0.1, center_momentum=0.9,
                 center_momentum_c=0.9, args=None):
        super().__init__()
        self.student_temp = student_temp
        self.student_temp_c = student_temp_c
        self.center_momentum = center_momentum
        self.center_momentum_c = center_momentum_c
        self.ncrops = ncrops
        self.centroids_counter = torch.tensor(0, device='cuda')
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_c", torch.zeros(1, out_dim_c))
        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp_schedule_c = np.concatenate((
            np.linspace(warmup_teacher_temp_c, teacher_temp_c, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp_c
        ))
        self.args = args

    def forward(self, student_output, teacher_output, epoch, student_centroids, teacher_centroids,
                bootstrap_centroid_mask, teacher_nn_centroids, centroids_indices_knn, teacher_nn2_centroids,
                centroids_indices_knn2, bootstrap_centroid_mask2):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        with torch.autocast(device_type='cuda', enabled=False):
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.ncrops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(2)

            loss_d = 0
            n_loss_terms = 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    loss_d += loss.mean()
                    n_loss_terms += 1
            loss_d /= n_loss_terms

            # Compute the loss on the centroids if provided
            loss_c = torch.tensor(0., device=student_output.device)

            # Sharpen the student predictions
            student_cent = student_centroids / self.student_temp_c

            # Teacher centering and sharpening
            temp = self.teacher_temp_schedule_c[epoch]
            teacher_cent = F.softmax((teacher_centroids - self.center_c) / temp, dim=-1)
            teacher_nn_centroids = F.softmax((teacher_nn_centroids - self.center_c) / temp, dim=-1)
            teacher_nn2_centroids = F.softmax((teacher_nn2_centroids - self.center_c) / temp, dim=-1)

            # Split the centroids view-wise
            student_cent_v1, student_cent_v2 = student_cent.chunk(2)
            teacher_cent_v1, teacher_cent_v2 = teacher_cent.chunk(2)
            teacher_cent_nn_v1, teacher_cent_nn_v2 = teacher_nn_centroids.unbind()
            teacher_cent_nn2_v1, teacher_cent_nn2_v2 = teacher_nn2_centroids.unbind()

            # Compute the loss with NNs
            loss_nn = torch.tensor(0., device=student_output.device)
            if epoch >= 1:
                nb_bootstrap = bootstrap_centroid_mask.sum()
                count = 0
                if nb_bootstrap > 0:
                    loss_nn += torch.sum(-teacher_cent_nn_v1 * F.log_softmax(student_cent_v2, dim=-1), dim=-1)[
                                   bootstrap_centroid_mask].sum() / nb_bootstrap
                    loss_nn += torch.sum(-teacher_cent_nn2_v2 * F.log_softmax(student_cent_v1, dim=-1), dim=-1)[
                                   bootstrap_centroid_mask2].sum() / nb_bootstrap
                    count += 2
                    loss_nn /= count

            # Compute the loss with views
            loss_c += torch.sum(-teacher_cent_v1 * F.log_softmax(student_cent_v2, dim=-1), dim=-1).mean()
            loss_c += torch.sum(-teacher_cent_v2 * F.log_softmax(student_cent_v1, dim=-1), dim=-1).mean()
            loss_c /= 2

            # Update the centers
            self.update_center(teacher_output, teacher_centroids)
            return loss_d, loss_c, loss_nn

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_centroids=None):
        """
        Update center used for teacher output.
        """
        # Image-level
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        # Update
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        # Centroids-level
        if teacher_centroids is not None:
            batch_center_c = torch.sum(teacher_centroids, dim=0, keepdim=True)
            self.centroids_counter = torch.tensor(len(teacher_centroids), device='cuda')

            # Update
            dist.all_reduce(batch_center_c)
            dist.all_reduce(self.centroids_counter)
            batch_center_c = batch_center_c / self.centroids_counter
            self.center_c = self.center_c * self.center_momentum_c + batch_center_c * (1 - self.center_momentum_c)


class DataAugmentationCrIBo(object):
    def __init__(self, global_crops_scale, same_teacher_augmentations):
        self.same_teacher_augmentations = same_teacher_augmentations
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Spatial transformation
        self.spatial_transfo = MyCompose([
            RandomResizedCropWithPos(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            MyComposeInner([RandomHorizontalFlipWithFlipBool(p=0.5)]),
        ])

        # Color transformations
        self.color_transfo1 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.color_transfo2 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

    def __call__(self, image):
        mask = None
        if isinstance(image, tuple):
            image, mask = image

        # Apply the spatial transformations
        view_1_, pos_1, mask_1 = self.spatial_transfo((image, mask))
        view_2_, pos_2, mask_2 = self.spatial_transfo((image, mask))

        view_1, view_2 = self.color_transfo1(view_1_), self.color_transfo2(view_2_)
        crops = [view_1, view_2]
        crops_pos = [pos_1, pos_2]
        if mask is None:
            return crops, crops_pos
        crops_mask = [mask_1, mask_2]
        return crops, crops_pos, crops_mask


class Clustering:
    def __init__(self, args, n_tokens, n_heads, sinkhorn_lambda, sinkhorn_iterations=3, student_temp=1.,
                 pos_alpha=(0.2, 0.2)):
        self.patch_size = args.patch_size
        self.n_tokens = n_tokens
        self.student_temp = student_temp
        self.pos_alpha = np.linspace(pos_alpha[1], pos_alpha[0], args.epochs)
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        self.n_heads = n_heads
        self.args = args

    @torch.no_grad()
    def sinkhorn(self, M, r, c, lambda_, iterations):
        P = torch.exp(- lambda_ * M).float()
        P /= reduce(P, 'b n k -> b 1 1', reduction='sum')

        # Iterate over the sinkhorn algorithm
        for _ in range(iterations):
            u = reduce(P, 'b n k -> b n 1', reduction='sum')
            P *= (r / u)
            u = reduce(P, 'b n k -> b 1 k', reduction='sum')
            P *= (c / u)
        P = torch.nan_to_num(P, nan=1e-8)
        return P, torch.sum(P * M, dim=[1, 2])

    def compute_assignments(self, tokens, positions, k, pos_alpha, sinkhorn_lambda, sinkhorn_iterations,
                            use_hard_assignment=True):
        # Normalize the tokens
        tokens = F.normalize(tokens, dim=-1)

        # Get the dimensions
        b, n, d = tokens.shape

        # Compute the random distribution
        r_uni = (torch.ones([b, n, 1], device=self.args.gpu) / n)
        r = r_uni
        c = (torch.ones([b, 1, k], device=self.args.gpu) / k)
        p = r_uni.squeeze()
        index = p.multinomial(num_samples=k, replacement=False)
        index = rearrange(index, 'b k -> (b k)')
        index = torch.eye(n, device=index.device)[index].to(tokens.device)
        index = rearrange(index, '(b k) n -> b k n', b=b)

        # Set the initial centroids
        centroids = torch.einsum('b n d, b k n -> b k d', tokens, index)

        assignment = index.permute(0, 2, 1)

        for _ in range(self.args.n_iter):
            # Compute the semantic similarity
            sem_similarity = torch.einsum('b n d, b k d -> b n k', tokens, centroids)

            # Compute the distance matrix
            pos_similarity = torch.sqrt(torch.sum((positions[:, None, :, :] - positions[:, :, None, :]) ** 2, dim=-1))
            pos_similarity = torch.einsum('B N n, B n k -> B N n k', pos_similarity, assignment)

            tmp = torch.ones_like(pos_similarity)
            tmp[pos_similarity == 0.0] = 0.0
            tmp = tmp.sum(dim=2, keepdim=True)

            pos_similarity[torch.logical_and(pos_similarity == 0.0,
                                             tmp != 0.0)] = 1e5  # If column is not zero, replace all 0 values with high value
            pos_similarity = einops.reduce(pos_similarity, 'B N n k -> B N k', reduction='min')

            # If cost is 0, replace with average cost
            avg_cost = pos_similarity.mean(dim=[1, 2], keepdim=True)
            avg_cost = repeat(avg_cost, 'b 1 1 -> b n k', k=k, n=n)
            pos_similarity[pos_similarity == 0.0] = avg_cost[pos_similarity == 0.0]
            pos_similarity /= pos_similarity.amax(dim=(-1, -2))[:, None, None]

            # Get the cost
            M = - sem_similarity + pos_alpha * pos_similarity
            M = (M - M.min()) / (M.max() - M.min())

            # Compute the transportation plan and the distance
            assignment, cost = self.sinkhorn(
                M=M,
                r=r,
                c=c,
                lambda_=sinkhorn_lambda,
                iterations=sinkhorn_iterations
            )

            # Compute the hard assignments
            hard_assignment = torch.max(assignment, dim=-1, keepdim=True).values
            hard_assignment = repeat(hard_assignment, 'b n 1 -> b n k', k=k)
            hard_assignment = (assignment == hard_assignment).float()

            if use_hard_assignment:
                assignment = hard_assignment

            # Update c
            if self.args.update_c:
                c = hard_assignment.sum(dim=1, keepdim=True) + 1e-2
                c /= c.sum(dim=-1, keepdim=True)

            # Update the centroids
            centroids = torch.einsum('b n d, b n k -> b k d', tokens, assignment)
            centroids = F.normalize(centroids, dim=-1)

        # Normalize column-wise and view-wise
        assignment = rearrange(assignment, 'b (m n) k -> m b n k', m=2)
        assignment_v1, assignment_v2 = assignment.unbind()

        # Normalize hard assignment
        # If a cluster is not present in two views, the normalization will divide by 0
        # If that happens, we just replace the 0 by 1
        # Later on, the centroids originating from that cluster will be discarded anyways
        tmpv1 = assignment_v1.sum(dim=-2, keepdim=True)
        tmpv2 = assignment_v2.sum(dim=-2, keepdim=True)
        tmpv1[tmpv1 == 0.0] = 1.0
        tmpv2[tmpv2 == 0.0] = 1.0

        assignment_v1 = assignment_v1 / tmpv1
        assignment_v2 = assignment_v2 / tmpv2
        assignment = torch.cat([assignment_v1, assignment_v2], dim=1)
        return assignment, cost, index

    def compute_student_centroids(self, assignments, tokens, valid_centroids):
        # Reshape the tokens
        tokens = rearrange(tokens[:, 1:], '(m b) n d -> m b n d', m=2)

        # Compute the centroids
        centroids = torch.einsum('m b n d, m b h n k -> m b h k d', tokens, assignments)
        centroids = rearrange(centroids, 'm b h k d -> m (b h k) d')

        # Split the centroids view-wise
        centroids_v1, centroids_v2 = centroids.unbind()
        centroids_v1, centroids_v2 = centroids_v1[valid_centroids], centroids_v2[valid_centroids]
        return centroids_v1, centroids_v2

    def compute_teacher_centroids(self, last_tokens, tokens, positions, epoch, use_hard_assignment=True):
        with torch.autocast(device_type='cuda', enabled=False):
            # Discard the [CLS] token
            tokens = tokens[:, 1:]
            last_tokens = last_tokens[:, 1:]
            tokens = rearrange(tokens, '(m b h) n d -> (b h) (m n) d', m=2, h=self.n_heads)
            last_tokens = rearrange(last_tokens, '(m b) n d -> m b n d', m=2)

            # Compute area of views
            tmp = positions[:, :, :, [0, -1]]
            diff = torch.abs(tmp[:, :, :, 0] - tmp[:, :, :, 1])
            area = torch.prod(diff, dim=-1)

            # Patchify positional encodings
            positions = rearrange(positions, "m b d n -> b (m n) d")
            positions = repeat(positions, 'b n d -> (b h) n d', h=self.n_heads)

            # Compute the assignments 
            assignments, _, _ = self.compute_assignments(tokens, positions, self.n_tokens, self.pos_alpha[epoch],
                                                         self.sinkhorn_lambda, self.sinkhorn_iterations,
                                                         use_hard_assignment)

            # ============================ Split the cluster view-wise ===========================================
            # Each token belongs to a single cluster
            hard_assignments = torch.max(assignments, dim=-1, keepdim=True).values
            hard_assignments = repeat(hard_assignments, 'b n 1 -> b n k', k=assignments.shape[-1])
            hard_assignments = (assignments == hard_assignments).float()
            hard_assignments = rearrange(hard_assignments, 'b (m n) k -> m b n k', m=2)
            assignments = rearrange(assignments, '(b h) (m n) k -> m b h n k', m=2, h=self.n_heads)

            # Compute the centroids of each view and normalize the assignments
            centroids = torch.einsum('m b n d, m b h n k -> m b h k d', last_tokens, assignments)
            centroids = rearrange(centroids, 'm b h k d -> m (b h k) d')
            centroids_v1, centroids_v2 = centroids.unbind()

            # Discard a cluster if it's empty in either view
            hard_assignments_v1, hard_assignments_v2 = rearrange(hard_assignments, 'm b n k -> m (b k) n').unbind()
            valid_centroids = torch.logical_and((hard_assignments_v1.sum(dim=-1) > 0),
                                                (hard_assignments_v2.sum(dim=-1) > 0))
            centroids_v1, centroids_v2 = centroids_v1[valid_centroids], centroids_v2[valid_centroids]

            # Correct the number of centroids per view
            centroids_per_view = torch.tensor_split(valid_centroids, tokens.shape[0])
            centroids_per_view = torch.stack([cpv.sum() for cpv in centroids_per_view])

            # Count the average number of regions
            region_count = centroids_per_view.float().mean().item()
            return (centroids_v1, centroids_v2), valid_centroids, assignments, region_count, centroids_per_view


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.epochs <= 2:
        args.warmup_teacher_temp_epochs = 0
        args.warmup_epochs = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_cribo(args)
