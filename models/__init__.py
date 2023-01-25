import os
import sys

import torch
import torchvision
import torch.nn.functional as F

from .language_net import TextModel, ClipTextModel, ClipVitTextModel
from .synthesizer_net import InnerProd, Bias, UpsampleInnerProd
from .audio_net import Unet, SqueezeUnet, DebugSqueezeUnet, AudioVisual7layerUNet, DistillAudioVisual7layerUNet, FTLatentDistillAudioVisual7layerUNet, BimodalDistillAudioVisual7layerUNet
from .vision_net import ResnetOrig, ResnetFC, ResnetDilated, Resnet50Dilated, ClipVisual, ClipVisualResDilated, ClipVisualResOrig, ClipVisualResJoint, ClipVisualVitOrig, ClipVisualVitJoint, ClipVisualResDebug, ClipVisualVitDebug, ImageNetVit, ClipVisualResDistill, ClipVisualVitDistill, ImageNetVitJoint, ImageNetVitDistill, ClipVisualResJointDropout,ClipVisualResDetr, ClipJointAttn, ClipVisualResDistillAttn, LatentJointAttn, ClipVisualCombinedJoint
from .criterion import BCELoss, L1Loss, L2Loss, BCEDropoutLoss, L1DropoutLoss


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unknown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet5', fc_dim=64, weights='', args=None):
        # 2D models
        if arch == 'unet5':
            net_sound = Unet(fc_dim=fc_dim, num_downs=5)
        elif arch == 'unet6':
            net_sound = Unet(fc_dim=fc_dim, num_downs=6)
        elif arch == 'unet7':
            net_sound = Unet(fc_dim=fc_dim, num_downs=7)
        elif 'distill-bimodal-audiovisual7layerunet' in arch:
            net_sound = BimodalDistillAudioVisual7layerUNet(args)
        elif 'ft-latent-distill-audiovisual7layerunet' in arch:
            net_sound = FTLatentDistillAudioVisual7layerUNet(args)
        elif 'distill-audiovisual7layerunet' in arch:
            net_sound = DistillAudioVisual7layerUNet(args)
        elif arch == 'audiovisual7layerunet':
            net_sound = AudioVisual7layerUNet(args)
        elif 'debug-squeezeunet7' in arch:
            net_sound = DebugSqueezeUnet(args, arch, fc_dim=fc_dim, num_downs=7)            
        elif 'squeezeunet7' in arch:
            net_sound = SqueezeUnet(arch, fc_dim=fc_dim, num_downs=7)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound
        
    # builder for language
    def build_language(self, fc_dim=64):
        net_text = TextModel(fc_dim)
        return net_text
        
    # builder for language
    def build_clip_language(self, arch, fc_dim=64):
        if 'vitb32' in arch:
            net_text = ClipVitTextModel(arch, fc_dim)
        else:
            net_text = ClipTextModel(arch, fc_dim)
        return net_text

    # builder for vision
    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool',
                    weights='', args=None):
        pretrained=True
        
        if 'resnet18fc' in arch :
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetFC(
                arch, original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif 'resnet18orig' in arch:
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetOrig(
                arch, original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif 'resnet18dilated' in arch :
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                arch, original_resnet, fc_dim=fc_dim, pool_type=pool_type)            
        elif 'resnet50orig' in arch:
            original_resnet = torchvision.models.resnet50(pretrained)
            net = ResnetOrig(
                arch, original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet50dilated':
            original_resnet = torchvision.models.resnet50(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type, res_model='resnet50')
        elif arch == 'resnet50dilated-moco':
            original_resnet = torchvision.models.resnet50(False)
            net = Resnet50Dilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type, res_model='resnet50')
        elif arch == 'clip':
            net = ClipVisual(fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-distill-attn' in arch:
            net = ClipVisualResDistillAttn(args, arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-audio-distill' in arch:
            net = ClipVisualResDistill(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'clip-res50-dilated':
            net = ClipVisualResDilated(fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-debug' in arch:
           net = ClipVisualResDebug(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-orig' in arch:
           net = ClipVisualResOrig(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-dropout-joint' in arch:
           net = ClipVisualResJointDropout(args, arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-joint-combinedvis' in arch:
           net = ClipVisualCombinedJoint(args, arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-joint' in arch:
           net = ClipVisualResJoint(args, arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-distill-res50-joint' in arch:
            net = ClipVisualResJointDistill(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-vitb32-audio-distill' in arch:
            net = ClipVisualVitDistill(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-vitb32-debug' in arch:
           net = ClipVisualVitDebug(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-vitb32-orig' in arch:
           net = ClipVisualVitOrig(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-vitb32-joint' in arch:
           net = ClipVisualVitJoint(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'imagenet-vitb32-audio-distill' in arch:
           net = ImageNetVitDistill(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'imagenet-vitb32' in arch:
           net = ImageNetVit(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'imagenet-vit-joint' in arch:
           net = ImageNetVitJoint(arch=arch, fc_dim=fc_dim, pool_type=pool_type)
        elif 'clip-res50-avt-detr' in arch:
           net = ClipVisualResDetr(args, arch=arch, fc_dim=fc_dim, pool_type=pool_type, dim_feedforward = args.ffn_dim, num_layers=args.num_transformer_layers, num_attn_heads=args.num_attention_heads)
        elif 'visualize-clip' in arch:
           net = ClipJointAttn(arch)
        elif 'visualize-latent' in arch:
           net = LatentJointAttn(arch)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_synthesizer(self, arch, fc_dim=64, weights='', args=None):    
        if arch == 'linear':
            net = InnerProd(fc_dim=fc_dim)
        elif 'linear-upsample' in arch:
            net = UpsampleInnerProd(arch, fc_dim)
        elif arch == 'bias':
            net = Bias()
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init)
        
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'bce_dropout':
            net = BCEDropoutLoss()
        elif arch == 'l1_dropout':
            net = L1DropoutLoss()
        else:
            raise Exception('Architecture undefined!')
        return net
