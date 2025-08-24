#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import math

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "test.json"  # change to train.json when running on training set
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 25)  # 32 leads to run-out-of-memory
        self.max_epoch = 40
        self.print_interval = 20
        self.eval_interval = 1e5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 5

        self.cur_epoch = 0

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def get_moco_scale(self):
        return self.get_adjust_scale(t=1.0)

    def get_reid_scale(self):
        return self.get_adjust_scale(t=0.8)

    def get_adjust_scale(self, t=0.8):
        decay_thresh = self.max_epoch * t
        if self.cur_epoch <= decay_thresh:
            return (1 - math.cos(self.cur_epoch * math.pi / decay_thresh)) / 2  # 0 -> 1
        else:
            return 1.

    # def random_resize(self, data_loader, epoch, rank, is_distributed):
    #     # TODO: close multi-scale training
    #     return self.input_size


    # ----------------- Model -----------------
    def get_model(self, settings=None):
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead, MoCo

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            if settings is not None and isinstance(settings, dict) and "seq_names" in settings:
                moco = MoCo(backbone, head, settings["seq_names"], dim=head.emb_dim)
            else:
                moco = None
            self.model = YOLOX(backbone, head, moco=moco)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    # ------------- Helpers -------------
    def _split_dir_for_json(self, json_file: str) -> str:
        """Trả về 'images/test' nếu json chứa 'test', ngược lại 'images/train'."""
        jf = (json_file or "").lower()
        return "images/test" if "test" in jf else "images/train"

    def _normalize_filenames_for_name_join(self, dataset, name_dir: str):
        """
        pull_item() sẽ mở: os.path.join(data_dir, name, file_name)
        -> Đảm bảo file_name KHÔNG bắt đầu bằng 'images/...'
        (nếu có thì strip đi để tránh trùng lặp).
        """
        strip_prefixes = ("images/train/", "images/val/", "images/test/")
        # 1) coco.dataset["images"] + coco.imgs
        imgs = dataset.coco.dataset.get("images", [])
        for im in imgs:
            fn = im.get("file_name", "")
            for s in strip_prefixes:
                if fn.startswith(s):
                    im["file_name"] = fn[len(s):]
                    break
            img_id = im.get("id")
            if img_id in dataset.coco.imgs:
                dataset.coco.imgs[img_id]["file_name"] = im["file_name"]
        # 2) các cache list
        for attr in ["annotations", "data_list", "imgs_list", "image_info", "images", "img_list"]:
            if hasattr(dataset, attr) and isinstance(getattr(dataset, attr), list):
                lst = getattr(dataset, attr)
                for rec in lst:
                    if isinstance(rec, dict):
                        fn = rec.get("file_name", "")
                        for s in strip_prefixes:
                            if fn.startswith(s):
                                rec["file_name"] = fn[len(s):]
                                break
        # 3) map theo id (nếu có)
        for attr in ["id2img", "imgid2ann", "imgid2info"]:
            mp = getattr(dataset, attr, None)
            if isinstance(mp, dict):
                for _, rec in mp.items():
                    if isinstance(rec, dict):
                        fn = rec.get("file_name", "")
                        for s in strip_prefixes:
                            if fn.startswith(s):
                                rec["file_name"] = fn[len(s):]
                                break

    # ------------- Train loader -------------
    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            list_collate,
        )

        data_root = os.path.join(get_yolox_datadir(), "MOT17")
        name_dir = self._split_dir_for_json(self.train_ann)   # 'images/train' hoặc 'images/test'

        base_dataset = MOTDataset(
            data_dir=data_root,
            json_file=self.train_ann,
            name=name_dir,                    # <<< QUAN TRỌNG
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
            max_epoch=self.max_epoch,
            is_training=True,
        )

        # Đảm bảo file_name không lặp prefix 'images/...'
        self._normalize_filenames_for_name_join(base_dataset, name_dir)

        seq_names = base_dataset.seq_names
        img_path2seq = base_dataset.img_path2seq

        # Giữ wrapper để tránh lỗi trainer.dataset._dataset; tắt mosaic/mixup
        dataset = MosaicDetection(
            base_dataset,
            mosaic=False,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=False,
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=False,
        )

        train_loader = DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            collate_fn=list_collate,
        )

        settings = {"seq_names": seq_names, "img_path2seq": img_path2seq}
        return train_loader, settings

    # ------------- Infer loader (train set) -------------
    def get_infer_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform, list_collate

        data_root = os.path.join(get_yolox_datadir(), "MOT17")
        name_dir = self._split_dir_for_json(self.train_ann)   # 'images/train'

        traindataset = MOTDataset(
            data_dir=data_root,
            json_file=self.train_ann,
            img_size=self.test_size,
            name=name_dir,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                get_tgt=True,
            ),
        )
        self._normalize_filenames_for_name_join(traindataset, name_dir)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(traindataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(traindataset)

        infer_loader = torch.utils.data.DataLoader(
            traindataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            collate_fn=list_collate,
        )
        return infer_loader

    # ------------- Eval loader (val/test) -------------
    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        data_root = os.path.join(get_yolox_datadir(), "MOT17")
        # val_half.json vẫn dùng ảnh trong images/train
        name_dir = self._split_dir_for_json(self.val_ann)     # 'images/train' hoặc 'images/test'

        valdataset = MOTDataset(
            data_dir=data_root,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=name_dir,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        self._normalize_filenames_for_name_join(valdataset, name_dir)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        val_loader = torch.utils.data.DataLoader(
            valdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )
        return val_loader

    # ------------- Evaluator -------------
    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator