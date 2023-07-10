# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.utils.net_utils import _smooth_l1_loss
from model.utils.config import cfg
from model.roi_layers import ROIAlign, ROIPool
import pdb
from torchvision.utils import save_image
import numpy as np

class PMSF(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PMSF,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.avg_pool_2=nn.AvgPool2d(2,2,0)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(),
        )
        self.fc1=nn.Sequential(
            nn.Linear(channel // reduction, channel//2, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel // reduction, channel//2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self,x,y):
        b,c,_,_=x.size()
        x_=self.avg_pool(x).view(b,c)
        y_=self.avg_pool(y).view(b,c)

        common=self.fc(torch.cat([x_,y_],dim=1))
        x_w=self.fc1(common).view(b,c,1,1)
        y_w=self.fc2(common).view(b,c,1,1)

        return x*x_w+y_w*self.avg_pool_2(y)


class Mutual_Related(nn.Module):
    """ mutual related attention """

    def __init__(self, in_channels):
        super(Mutual_Related, self).__init__()

        self.globle_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_shared = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

        self.w_conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, thermal, rgb, pre_w=None):
        b, c, _, _ = thermal.size()
        v_t = self.globle_avg_pool(thermal).view(b, c)
        v_rgb = self.globle_avg_pool(rgb).view(b, c)

        mean_rgb = torch.mean(rgb, dim=1, keepdim=True)
        mean_t = torch.mean(thermal, dim=1, keepdim=True)
        cat_mean = torch.cat([mean_rgb, mean_t], dim=1)
        w_ori = self.w_conv(cat_mean)

        act_shared_t = self.conv_shared(v_t).view(b, c, 1, 1).expand_as(thermal)
        act_shared_rgb = self.conv_shared(v_rgb).view(b, c, 1, 1).expand_as(rgb)

        if pre_w == None:
            w_new = w_ori
        else:
            if pre_w.size(2) > w_ori.size(2):
                pre_w = F.max_pool2d(pre_w, kernel_size=(2, 2), stride=2)

            w_new = (w_ori + pre_w) / 2.0

        out_thermal = thermal + act_shared_rgb * rgb * w_new.expand_as(rgb)
        out_rgb = rgb + act_shared_t * thermal * w_new.expand_as(thermal)

        return out_thermal, out_rgb, w_ori




class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        # self.index = 1

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg_infrared = models.vgg16()
        vgg_color = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load( self.model_path)
            vgg_infrared.load_state_dict({k: v for k, v in state_dict.items() if k in vgg_infrared.state_dict()})
            vgg_color.load_state_dict({k: v for k, v in state_dict.items() if k in vgg_color.state_dict()})

        vgg_infrared.classifier = nn.Sequential(*list(vgg_infrared.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # self.infrared_conv4 = nn.Sequential(*list(vgg_infrared.features._modules.values())[:23])
        # self.color_conv4 = nn.Sequential(*list(vgg_color.features._modules.values())[:23])

        self.infrared_conv_1_1 = nn.Sequential(*list(vgg_infrared.features._modules.values())[:12])
        self.color_conv_1_1 = nn.Sequential(*list(vgg_color.features._modules.values())[:12])
        self.mutual_1_1 = Mutual_Related(256)

        self.infrared_conv_1_2 = nn.Sequential(*list(vgg_infrared.features._modules.values())[12:14])
        self.color_conv_1_2 = nn.Sequential(*list(vgg_color.features._modules.values())[12:14])
        self.mutual_1_2 = Mutual_Related(256)

        self.infrared_conv_1_3 = nn.Sequential(*list(vgg_infrared.features._modules.values())[14:16])
        self.color_conv_1_3 = nn.Sequential(*list(vgg_color.features._modules.values())[14:16])
        self.mutual_1_3 = Mutual_Related(256)

        self.infrared_conv_2_1 = nn.Sequential(*list(vgg_infrared.features._modules.values())[16:19])
        self.color_conv_2_1 = nn.Sequential(*list(vgg_color.features._modules.values())[16:19])
        self.mutual_2_1 = Mutual_Related(512)

        self.infrared_conv_2_2 = nn.Sequential(*list(vgg_infrared.features._modules.values())[19:21])
        self.color_conv_2_2 = nn.Sequential(*list(vgg_color.features._modules.values())[19:21])
        self.mutual_2_2 = Mutual_Related(512)

        self.infrared_conv_2_3 = nn.Sequential(*list(vgg_infrared.features._modules.values())[21:23])
        self.color_conv_2_3 = nn.Sequential(*list(vgg_color.features._modules.values())[21:23])

        self.fusion_base = nn.Sequential(*list(vgg_infrared.features._modules.values())[23:-1])

        self.mutual_conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.inner_conv_color = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.inner_conv_infrared = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.pmsf = PMSF(1024)
        self.RCNN_roi_pool_conv4 = ROIPool((cfg.POOLING_SIZE * 2, cfg.POOLING_SIZE * 2), 1.0 / 8.0)
        self.RCNN_roi_align_conv4 = ROIAlign((cfg.POOLING_SIZE * 2, cfg.POOLING_SIZE * 2), 1.0 / 8.0, 0)

        # Fix the layers before conv3:
        for layer in range(12):
            for p in self.infrared_conv_1_1[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_1_1[layer].parameters(): p.requires_grad = False

        for layer in range(2):
            for p in self.infrared_conv_1_2[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_1_2[layer].parameters(): p.requires_grad = False

        for layer in range(2):
            for p in self.infrared_conv_1_3[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_1_3[layer].parameters(): p.requires_grad = False

        for layer in range(3):
            for p in self.infrared_conv_2_1[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_2_1[layer].parameters(): p.requires_grad = False

        for layer in range(2):
            for p in self.infrared_conv_2_2[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_2_2[layer].parameters(): p.requires_grad = False

        for layer in range(2):
            for p in self.infrared_conv_2_3[layer].parameters(): p.requires_grad = False
            for p in self.color_conv_2_3[layer].parameters(): p.requires_grad = False

        for layer in range(7):
            for p in self.fusion_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg_infrared.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)


        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        infrared_im_data = im_data[0]
        color_im_data = im_data[1]

        if self.training:
            fake_color_img = torch.randn_like(color_im_data)
            fake_infrared_img = torch.randn_like(infrared_im_data)

            rand_num = np.random.uniform(0,1)
            if rand_num < 0.1:
                color_im_data = fake_color_img
            elif rand_num >0.9:
                infrared_im_data = fake_infrared_img



        batch_size = infrared_im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        color_feat = self.color_conv_1_1(color_im_data)
        infrared_feat = self.infrared_conv_1_1(infrared_im_data)
        infrared_feat, color_feat, w_1_1 = self.mutual_1_1(infrared_feat, color_feat)

        color_feat = self.color_conv_1_2(color_feat)
        infrared_feat = self.infrared_conv_1_2(infrared_feat)
        infrared_feat, color_feat, w_1_2 = self.mutual_1_2(infrared_feat, color_feat, w_1_1)

        color_feat = self.color_conv_1_3(color_feat)
        infrared_feat = self.infrared_conv_1_3(infrared_feat)
        infrared_feat, color_feat, w_1_3 = self.mutual_1_3(infrared_feat, color_feat, w_1_2)

        color_feat = self.color_conv_2_1(color_feat)
        infrared_feat = self.infrared_conv_2_1(infrared_feat)
        infrared_feat, color_feat, w_2_1 = self.mutual_2_1(infrared_feat, color_feat, w_1_3)

        color_feat = self.color_conv_2_2(color_feat)
        infrared_feat = self.infrared_conv_2_2(infrared_feat)
        infrared_feat, color_feat, w_2_2 = self.mutual_2_2(infrared_feat, color_feat, w_2_1)

        color_feat = self.color_conv_2_3(color_feat)
        infrared_feat = self.infrared_conv_2_3(infrared_feat)

        base_color_feat = self.fusion_base(color_feat)
        base_infrared_feat = self.fusion_base(infrared_feat)

        # feed base feature map tp RPN to obtain rois
        rois_color, rpn_loss_cls_color, rpn_loss_bbox_color = self.RCNN_rpn(base_color_feat, im_info, gt_boxes, num_boxes)
        rois_infrared, rpn_loss_cls_infrared, rpn_loss_bbox_infrared = self.RCNN_rpn(base_infrared_feat, im_info, gt_boxes,num_boxes)

        rois = torch.cat([rois_color,rois_infrared],dim=1)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            # pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_color = self.RCNN_roi_align(base_color_feat, rois.view(-1, 5))
            pooled_feat_infrared = self.RCNN_roi_align(base_infrared_feat, rois.view(-1, 5))

            pooled_feat_color_4 = self.RCNN_roi_align_conv4(color_feat, rois.view(-1,5))
            pooled_feat_infrared_4 = self.RCNN_roi_align_conv4(infrared_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            # pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            pooled_feat_color = self.RCNN_roi_pool(base_color_feat, rois.view(-1, 5))
            pooled_feat_infrared = self.RCNN_roi_pool(base_infrared_feat, rois.view(-1, 5))

            pooled_feat_color_4 = self.RCNN_roi_pool_conv4(color_feat, rois.view(-1, 5))
            pooled_feat_infrared_4 = self.RCNN_roi_pool_conv4(infrared_feat, rois.view(-1, 5))



        # first multiscale then fusion
        pooled_feat_color = self.pmsf(pooled_feat_color,pooled_feat_color_4)
        pooled_feat_infrared = self.pmsf(pooled_feat_infrared,pooled_feat_infrared_4)

        inner_weight_color = self.inner_conv_color(pooled_feat_color)
        pooled_feat_color_avg_pool = torch.mean(pooled_feat_color, dim=1, keepdim=True)
        pooled_feat_infrared_avg_pool = torch.mean(pooled_feat_infrared, dim=1, keepdim=True)
        feat_cat_avg_pool = torch.cat([pooled_feat_color_avg_pool, pooled_feat_infrared_avg_pool], dim=1)
        mutual_weight = self.mutual_conv(feat_cat_avg_pool)
        mutual_weight = inner_weight_color * mutual_weight
        inner_weight_infrared = self.inner_conv_infrared(pooled_feat_infrared)
        pooled_feat_infrared_weighted = (inner_weight_infrared + mutual_weight) * 0.5 * pooled_feat_infrared
        pooled_feat_color_weighted = pooled_feat_color * inner_weight_color
        pooled_feat = pooled_feat_color_weighted + pooled_feat_infrared_weighted
        pooled_feat = self._head_to_tail(pooled_feat)

        avg_weight_color = F.max_pool2d(inner_weight_color, 7).view(-1, 1)
        avg_weight_color = avg_weight_color.data
        avg_weight_infrared = F.max_pool2d((inner_weight_infrared + mutual_weight) * 0.5, 7).view(-1, 1)
        avg_weight_infrared = avg_weight_infrared.data

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce=False)
            avg_conf, _ = torch.max(torch.cat([avg_weight_color, avg_weight_infrared], dim=1), dim=1)
            RCNN_loss_cls = (RCNN_loss_cls * (1 - avg_conf)).mean()

            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) * 0.5 + RCNN_loss_cls * 0.5

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, reduce=False)
            RCNN_loss_bbox = RCNN_loss_bbox.mean() * 0.5 + (RCNN_loss_bbox * (1 - avg_conf)).mean() * 0.5

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rpn_loss_cls_color, rpn_loss_bbox_color, rpn_loss_cls_infrared, rpn_loss_bbox_infrared, RCNN_loss_cls, RCNN_loss_bbox
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label