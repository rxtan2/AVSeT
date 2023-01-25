import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import *
from .debug_clip_model import DebugModifiedResNet, DebugVisionTransformer
from .language_net import *
from .pytorch_pretrained_vit import ViT

class ResnetOrig(nn.Module):
    def __init__(self, arch, orig_resnet, fc_dim=32, pool_type='maxpool',
                 res_model='resnet18', dilate_scale=16, conv_size=3):
        super(ResnetOrig, self).__init__()
        from functools import partial
        self.arch = arch

        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])
            
        if 'finetune' not in self.arch:
            for k, v in self.features.named_parameters():
                v.requires_grad = False

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        
        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        if 'full-mean-center' in self.arch:
            x = torch.mean(x, dim=[-2, -1])
            x = x[:, 1, :]
        elif 'region-meanpool' in self.arch:
            x = x.permute(0, 2, 1, 3, 4)
            x = F.adaptive_avg_pool3d(x, 1)
            x = x.view(B, C)
        elif 'region-maxpool' in self.arch:        
            x = x.permute(0, 2, 1, 3, 4)
            x = F.adaptive_max_pool3d(x, 1)
            x = x.view(B, C)
        
        return x

class Resnet(nn.Module):
    def __init__(self, original_resnet):
        super(Resnet, self).__init__()
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1))
        return x


class ResnetFC(nn.Module):
    def __init__(self, arch, original_resnet, fc_dim=64,
                 pool_type='maxpool', conv_size=3):
        super(ResnetFC, self).__init__()
        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)
            
        if 'finetune' not in arch:
            for k, v in self.features.named_parameters():
                v.requires_grad = False

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)

        x = x.view(B, C)
        return x


class ResnetDilated(nn.Module):
    def __init__(self, arch, orig_resnet, fc_dim=64, pool_type='maxpool',
                 res_model='resnet18', dilate_scale=16, conv_size=3):
        super(ResnetDilated, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        if res_model=='resnet18':
            self.fc = nn.Conv2d(512, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif res_model=='resnet50':
            self.fc = nn.Conv2d(2048, fc_dim, kernel_size=conv_size, padding=conv_size//2)
            
        if 'finetune' not in arch:
            for k, v in self.features.named_parameters():
                v.requires_grad = False

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        
        return x
        
    def forward_center_frame(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)

        x = self.features(x)
               
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.squeeze(2)
        
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        #print('pool_type: ', self.pool_type)
        #print('input x: ', x.shape)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        #print('permute x: ', x.shape)
        
        x = x.view(B*T, C, H, W)
        
        #print('view x: ', x.shape)

        x = self.features(x)
        
        #print('x: ', x.shape)
        
        x = self.fc(x)
        
        #print('fc x: ', x.shape)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        #print('view fc x: ', x.shape)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        #print('view permute x: ', x.shape)
        #sys.exit()

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
            
        #print('pool x: ', x.shape)

        x = x.view(B, C)
        
        #print('final x: ', x.shape)
        #sys.exit()
        
        '''pool_type:  maxpool
        input x:  torch.Size([4, 3, 3, 224, 224])
        permute x:  torch.Size([4, 3, 3, 224, 224])
        view x:  torch.Size([12, 3, 224, 224])
        feat x:  torch.Size([12, 512, 14, 14])
        fc x:  torch.Size([12, 32, 14, 14])
        view fc x:  torch.Size([4, 3, 32, 14, 14])
        view permute x:  torch.Size([4, 32, 3, 14, 14])
        pool x:  torch.Size([4, 32, 1, 1, 1])
        final x:  torch.Size([4, 32])'''
        
        return x
        
class Resnet50Dilated(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool',
                 res_model='resnet50', dilate_scale=16, conv_size=3):
        super(Resnet50Dilated, self).__init__()
        from functools import partial
        
        pretrained_path = "/research/rxtan/object-detection/models/Sound-of-Pixels/initialization_weights/moco/moco_v2_800ep_pretrain.pth.tar"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/Sound-of-Pixels/initialization_weights/moco/moco_v2_800ep_pretrain.pth.tar"
        pretrained_dict = torch.load(pretrained_path)['state_dict']
        
        model_dict = orig_resnet.state_dict()
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            new_k = k.replace('module.encoder_q.', '')
            if new_k in model_dict:
                updated_pretrained_dict[new_k] = v
                
        model_dict.update(updated_pretrained_dict)
        #orig_resnet.load_state_dict(model_dict)

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(2048, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        
        return x
        
    def forward_center_frame(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.squeeze(2)
        
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        #print('pool_type: ', self.pool_type)
        #print('input x: ', x.shape)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        #print('permute x: ', x.shape)
        
        x = x.view(B*T, C, H, W)
        
        #print('view x: ', x.shape)

        x = self.features(x)
        
        #print('feat x: ', x.shape)
        
        x = self.fc(x)
        
        #print('fc x: ', x.shape)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        #print('view fc x: ', x.shape)
        
        x = x.permute(0, 2, 1, 3, 4)  
        
        #print('view permute x: ', x.shape)
        #sys.exit()

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
            
        #print('pool x: ', x.shape)

        x = x.view(B, C)
        
        #print('final x: ', x.shape)
        #sys.exit()
        
        return x
        
class ClipVisualResJoint(nn.Module):
    def __init__(self, args, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(ClipVisualResJoint, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        #self.logit_scale = state_dict['logit_scale']
        self.logit_scale = nn.Parameter(state_dict['logit_scale']) # use this for new frame ablation jobs !!!!!
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        
        self.scale = 'scale' in self.arch
        
        if 'post-fc' in self.arch:
            self.text_fc = nn.Linear(text_dim, fc_dim)
            if '-linear' in self.arch:
                self.vis_fc = nn.Linear(vis_dim, fc_dim)
            else:
                self.vis_fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif 'pre-fc' in self.arch:
            self.vis_fc = nn.Linear(vis_dim, fc_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        if 'attn' in self.arch:
            self.multihead_attn = nn.MultiheadAttention(1024, args.num_attention_heads, batch_first=True)
        elif 'trans' in self.arch:
            encoder_layer = nn.TransformerEncoderLayer(1024, args.num_attention_heads, args.ffn_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)
            
            self.selected_out = OrderedDict()
            self.intermediate_results = []
            
            self.intermediate_results.append(getattr(self.transformer_encoder._modules['layers']._modules[str(args.num_transformer_layers-1)], 'self_attn').register_forward_hook(self.forward_hook('self_attn')))
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
            self.logit_scale.requires_grad = False
        elif 'finetune' in self.arch and 'traintext' in self.arch:
            for k, v in self.visual.named_parameters():            
                if 'attnpool.c_proj' not in k and 'attnpool.v_proj' not in k:
                    v.requires_grad = False
                    
        self.logit_scale.requires_grad = False
                    
        for module in self.visual.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, vis, text, num_frames):
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = vis / vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)
        
        # Test runs with CLIP scaling factor
        if self.scale:
            logit_scale = self.logit_scale.exp()    
            scores = logit_scale * scores
            
        if 'spatiotemporal' in self.arch:
            scores = self.softmax(scores)
            if 'normalize-sum' in self.arch:
                agg_vis = norm_vis * scores.unsqueeze(-1)
            else:
                agg_vis = vis * scores.unsqueeze(-1)
            agg_vis = agg_vis.sum(1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = self.softmax(scores)
            
            if 'normalize-sum' in self.arch:
                norm_vis = norm_vis.view(norm_vis.size(0), num_frames, -1, norm_vis.size(-1))
                agg_vis = norm_vis * scores.unsqueeze(-1)
            else:
                vis = vis.view(vis.size(0), num_frames, -1, vis.size(-1))
                agg_vis = vis * scores.unsqueeze(-1)
                
            agg_vis = agg_vis.sum(-2)
            
            if 'maxpool' in self.arch:
                agg_vis = torch.max(agg_vis, dim=1)[0]
            else:
                agg_vis = agg_vis.mean(1)
                
            #scores = scores.view(scores.size(0), -1)
        
        return agg_vis, scores # scores = (B, 147)
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x, self.arch)
        x = x[:, 1:, :]
        
        spatial_dim = int(x.size(1) ** 0.5)
        
        if 'attn' in self.arch:
            x = torch.reshape(x, (B, T * x.size(1), x.size(-1)))
            num_regions = x.size(1)
            text = text.unsqueeze(1)
            norm_x = x / x.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            norm_seq = torch.cat([norm_text, norm_x], dim=1)
            seq = torch.cat([text, x], dim=1)
            agg_x, scores = self.multihead_attn(norm_seq, norm_seq, seq)
            agg_x = agg_x[:, 0, :]
            scores = scores[:, 0, 1:]
            scores = scores.view(scores.size(0), T, spatial_dim, spatial_dim)
            return agg_x, scores
        elif 'trans' in self.arch:
        
            x = torch.reshape(x, (B, T * x.size(1), x.size(-1)))
            text = text.unsqueeze(1)
            norm_x = x / x.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            norm_seq = torch.cat([norm_text, norm_x], dim=1)
            norm_seq = norm_seq.permute(1, 0, 2).contiguous()
            norm_seq = self.transformer_encoder(norm_seq)
            norm_seq = norm_seq.permute(1, 0, 2).contiguous()
            agg_x = norm_seq[:, 0, :]
            
            attn_head_output = self.selected_out['self_attn']
            visual_attn_weights = attn_head_output[1]
            visual_attn_weights = visual_attn_weights[:, 0, 1:]
            scores = visual_attn_weights.view(B, T, spatial_dim, spatial_dim)
            self.selected_out.clear()
         
            return agg_x, scores
        
        if 'post-fc' in self.arch:
        
            if '-linear' in self.arch:
                x = self.vis_fc(x)
                x = x.view(B, -1, x.size(-1))
            else:
                x = x.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
                x = self.vis_fc(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, (B, -1, x.size(-1)))
                
            text = self.text_fc(text)
        elif 'pre-fc' in self.arch:
            x = torch.reshape(x, (B, -1, x.size(-1)))
        else:
            # extract region-level representations after text projection
            x = torch.reshape(x, (B, -1, x.size(-1)))
            
        agg_x, scores = self.compute_agg_visual(x, text, T)
        scores = scores.view(B, T, spatial_dim, -1)
        
        if 'pre-fc' in self.arch:
            agg_x = self.vis_fc(agg_x)
        
        return agg_x, scores
        
class ClipVisualResJointDropout(nn.Module):
    def __init__(self, args, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(ClipVisualResJointDropout, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        #self.logit_scale = state_dict['logit_scale']
        self.logit_scale = nn.Parameter(state_dict['logit_scale']) # use this for new frame ablation jobs !!!!!
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        self.language_dropout = args.language_dropout
        
        self.scale = 'scale' in self.arch
        
        if 'post-fc' in self.arch:
            self.text_fc = nn.Linear(text_dim, fc_dim)
            if '-linear' in self.arch:
                self.vis_fc = nn.Linear(vis_dim, fc_dim)
            else:
                self.vis_fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif 'pre-fc' in self.arch:
            self.vis_fc = nn.Linear(vis_dim, fc_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        if 'attn' in self.arch:
            self.multihead_attn = nn.MultiheadAttention(1024, args.num_attention_heads, batch_first=True)
        elif 'trans' in self.arch:
            encoder_layer = nn.TransformerEncoderLayer(1024, args.num_attention_heads, args.ffn_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)
            
            self.selected_out = OrderedDict()
            self.intermediate_results = []
            
            self.intermediate_results.append(getattr(self.transformer_encoder._modules['layers']._modules[str(args.num_transformer_layers-1)], 'self_attn').register_forward_hook(self.forward_hook('self_attn')))
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
            self.logit_scale.requires_grad = False
        elif 'finetune' in self.arch and 'traintext' in self.arch:
            for k, v in self.visual.named_parameters():            
                if 'attnpool.c_proj' not in k and 'attnpool.v_proj' not in k:
                    v.requires_grad = False
                    
        self.logit_scale.requires_grad = False
                    
        for module in self.visual.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, vis, text, num_frames):
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = vis / vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)
        
        # Test runs with CLIP scaling factor
        if self.scale:
            logit_scale = self.logit_scale.exp()    
            scores = logit_scale * scores
            
        if 'spatiotemporal' in self.arch:
            scores = self.softmax(scores)
            if 'normalize-sum' in self.arch:
                agg_vis = norm_vis * scores.unsqueeze(-1)
            else:
                agg_vis = vis * scores.unsqueeze(-1)
            agg_vis = agg_vis.sum(1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = self.softmax(scores)
            
            if 'normalize-sum' in self.arch:
                norm_vis = norm_vis.view(norm_vis.size(0), num_frames, -1, norm_vis.size(-1))
                agg_vis = norm_vis * scores.unsqueeze(-1)
            else:
                vis = vis.view(vis.size(0), num_frames, -1, vis.size(-1))
                agg_vis = vis * scores.unsqueeze(-1)
                
            agg_vis = agg_vis.sum(-2)
            if 'maxpool' in self.arch:
                agg_vis = torch.max(agg_vis, dim=1)[0]
            else:
                agg_vis = agg_vis.mean(1)
            scores = scores.view(scores.size(0), -1)
        
        return agg_vis, scores # scores = (B, 147)
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, C, T, H, W) = x.size()
        
        dropout_probs = torch.rand(B).cuda()
        
        x = x.view(B*T, C, H, W)

        x = self.visual(x, self.arch)
        
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        
        if 'attn' in self.arch:
            x = torch.reshape(x, (B, T * x.size(1), x.size(-1)))
            num_regions = x.size(1)
            text = text.unsqueeze(1)
            norm_x = x / x.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            norm_seq = torch.cat([norm_text, norm_x], dim=1)
            seq = torch.cat([text, x], dim=1)
            agg_x, scores = self.multihead_attn(norm_seq, norm_seq, seq)
            agg_x = agg_x[:, 0, :]
            scores = scores[:, 0, 1:]
            scores = scores.view(scores.size(0), T, spatial_dim, spatial_dim)
            return agg_x, scores
        elif 'trans' in self.arch:
        
            x = torch.reshape(x, (B, T * x.size(1), x.size(-1)))
            text = text.unsqueeze(1)
            norm_x = x / x.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            norm_seq = torch.cat([norm_text, norm_x], dim=1)
            norm_seq = norm_seq.permute(1, 0, 2).contiguous()
            norm_seq = self.transformer_encoder(norm_seq)
            norm_seq = norm_seq.permute(1, 0, 2).contiguous()
            agg_x = norm_seq[:, 0, :]
            
            attn_head_output = self.selected_out['self_attn']
            visual_attn_weights = attn_head_output[1]
            visual_attn_weights = visual_attn_weights[:, 0, 1:]
            scores = visual_attn_weights.view(B, T, spatial_dim, spatial_dim)
            self.selected_out.clear()
         
            return agg_x, scores
        
        if 'post-fc' in self.arch:
        
            if '-linear' in self.arch:
                x = self.vis_fc(x)
                x = x.view(B, -1, x.size(-1))
            else:
                x = x.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
                x = self.vis_fc(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, (B, -1, x.size(-1)))
                
            text = self.text_fc(text)
        elif 'pre-fc' in self.arch:
            x = torch.reshape(x, (B, -1, x.size(-1)))
        else:
            # extract region-level representations after text projection
            x = torch.reshape(x, (B, -1, x.size(-1)))
            
        agg_x, scores = self.compute_agg_visual(x, text, T)
        scores = scores.view(B, T, spatial_dim, -1)
        
        x = x.view(x.size(0), x.size(-1), T, spatial_dim, -1)
        vis_x = F.adaptive_max_pool3d(x, 1)
        vis_x = vis_x.squeeze()

        dropout_probs = dropout_probs <= self.language_dropout
        
        if 'pre-fc' in self.arch:
            agg_x = self.vis_fc(agg_x)
        
        return agg_x, scores, vis_x, dropout_probs
        
class ClipVisualResJointDistill(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024):
        super(ClipVisualResJointDistill, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        self.clip_region_projection = nn.Linear(text_dim, text_dim)
        
        self.arch = arch
        
        if 'pretrained-unnorm' not in self.arch:
            pretrained_path = "/research/rxtan/object-detection/models/finetune_clip_visual/ckpt/clip_res50_linear_lr_0.0001/epoch0020.pth.tar"
            if not os.path.exists(pretrained_path):
                pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/finetune_clip_visual/ckpt/clip_res50_linear_lr_0.0001/epoch0020.pth.tar"
        else:
            pretrained_path = "/research/rxtan/object-detection/models/finetune_clip_visual/ckpt/unnormalized_clip_res50_linear_lr_0.0001/epoch0020.pth.tar"
            if not os.path.exists(pretrained_path):
                pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/finetune_clip_visual/ckpt/unnormalized_clip_res50_linear_lr_0.0001/epoch0020.pth.tar"
            
        pretrained_dict = torch.load(pretrained_path)['state_dict']
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        if 'post-fc' in self.arch:
            self.text_fc = nn.Linear(text_dim, fc_dim)
            if '-linear' in self.arch:
                self.vis_fc = nn.Linear(vis_dim, fc_dim)
            else:
                self.vis_fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif 'pre-fc' in self.arch:
            self.vis_fc = nn.Linear(vis_dim, fc_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
          
    def compute_agg_visual(self, vis, text):
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = vis / vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)
        scores = self.softmax(scores)
        agg_vis = vis * scores.unsqueeze(-1)
        agg_vis = agg_vis.sum(1)
        
        return agg_vis, scores
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)

        x = self.visual(x, self.arch)
        
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        
        x = self.clip_region_projection(x)
        
        if 'post-fc' in self.arch:
        
            if '-linear' in self.arch:
                x = x.permute(0, 2, 3, 1)
                x = self.vis_fc(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
                x = self.vis_fc(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, (B, -1, x.size(-1)))
                
            text = self.text_fc(text)
        elif 'pre-fc' in self.arch:
            x = torch.reshape(x, (B, -1, x.size(-1)))
            #agg_x, scores = self.compute_agg_visual(x, text)
        else:
        
            # normalized features
            #x = x / x.norm(dim=-1, keepdim=True)
            #text = text / text.norm(dim=-1, keepdim=True)

            # extract region-level representations after text projection
            x = torch.reshape(x, (B, -1, x.size(-1)))

        agg_x, scores = self.compute_agg_visual(x, text)
        scores = scores.view(B, T, spatial_dim, -1)
        
        return agg_x, scores
        
class ClipVisualResOrig(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualResOrig, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        orig_resnet = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = orig_resnet.state_dict()
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            new_k = k.replace('visual.', '')
            if new_k in model_dict:
                updated_pretrained_dict[new_k] = v
                
        model_dict.update(updated_pretrained_dict)
        
        if 'random' not in self.arch:
            orig_resnet.load_state_dict(model_dict)

        if 'full-mean' in self.arch:
            self.features = nn.Sequential(*list(orig_resnet.children())[:])
        elif 'region' in self.arch:
        
            if '-top' in self.arch:
                self.features = nn.Sequential(*list(orig_resnet.children())[:])
            else:
                self.features = nn.Sequential(*list(orig_resnet.children())[:-2])
                
            
        # Freeze clip
        for k, v in self.features.named_parameters():
          v.requires_grad = False
          
        if '-conv' in self.arch:
            self.fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif '-fc' in self.arch:
            self.fc = nn.Linear(1024, fc_dim)
            
    def forward_multiframe(self, x, pool=True):
        
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)
        
        x = self.features(x)
        
        if 'full-mean-center' in self.arch:
        
            x = x[:, 0, :]
            x = x.view(B, T, -1)
            x = x[:, 1, :]

        elif 'full-mean' in self.arch:
            x = x[:, 0, :]
            
            x = x.view(B, T, -1)
            x = x.mean(1)
            
            if '-fc' in self.arch:
                x = self.fc(x)
            
        elif 'region' in self.arch:        
            if '-top' in self.arch:
                x = x[:, 1:, :]
            
                x = x.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), 7, 7)
        
            if '-conv' in self.arch:
                x = self.fc(x)
            elif '-fc' in self.arch:            
                x = x.permute(0, 2, 3, 1)
                x = self.fc(x)
                x = x.permute(0, 3, 1, 2)
        
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            
            if 'maxpool' in self.arch:
                x = F.adaptive_max_pool3d(x, 1)
            elif 'meanpool' in self.arch:
                x = F.adaptive_avg_pool3d(x, 1)           
            x = x.view(B, C)
        
        return x
        
class ClipVisualResDebug(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualResDebug, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = DebugModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                arch=self.arch
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
            
        if 'finetune' not in self.arch:            
            # Freeze clip
            for k, v in self.named_parameters():
                v.requires_grad = False
            
    def forward_multiframe(self, x, pool=True):   
     
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x, self.arch)
        
        if 'contextnocls' not in self.arch and 'layer4' not in self.arch:
            cls_x = x[:, 0, :]
            x = x[:, 1:, :]
            
        if 'normalize' in self.arch:
            x = F.normalize(x, dim=-1)
            
        spatial_dim = int(x.size(1) ** 0.5)
        
        x = x.view(B, T, spatial_dim, spatial_dim, x.size(-1))
        x = x.permute(0, 4, 1, 2, 3)
        
        if 'maxpool' in self.arch:
            x = F.adaptive_max_pool3d(x, 1)
        elif 'meanpool' in self.arch:
            x = F.adaptive_avg_pool3d(x, 1)
            
        x = x.view(B, -1)
        
        '''if 'full-mean-center-layer4' in self.arch or 'full-mean-center-layer3' in self.arch:
            x = x.view(x.size(0), x.size(1), -1)
            x = x.permute(0, 2, 1)
            x = x.view(B, T, x.size(-2), x.size(-1))
            x = x[:, 1, :, :]
            x = x.mean(1)
        elif 'full-mean' in self.arch:
            x = x[:, 0, :]
            x = x.view(B, T, -1)
            
            if 'center' in self.arch:
                x = x[:, 1, :]
            else:
                x = x.mean(1)
        elif 'region' in self.arch:
        
            if ('layer4' in self.arch or 'layer3' in self.arch) and 'res50' in self.arch:
                (_, C, H, W) = x.size()
                x = x.view(B, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)
            else:
                (_, _, C) = x.size()
                x = x[:, 1:, :]
                spatial_dim = int(x.size(1) ** 0.5)
                x = x.view(B, T, spatial_dim, spatial_dim, -1)
                x = x.permute(0, 4, 1, 2, 3)
            
            if 'maxpool' in self.arch:
                x = F.adaptive_max_pool3d(x, 1)
            elif 'meanpool' in self.arch:
                x = F.adaptive_avg_pool3d(x, 1)
                
            x = x.view(B, C)'''
        
        return x

class ClipVisualResDilated(nn.Module):
    def __init__(self, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualResDilated, self).__init__()
        from functools import partial
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        orig_resnet = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = orig_resnet.state_dict()
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            new_k = k.replace('visual.', '')
            if new_k in model_dict:
                updated_pretrained_dict[new_k] = v
                
        model_dict.update(updated_pretrained_dict)
        orig_resnet.load_state_dict(model_dict)
        
        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
        
        self.features = nn.Sequential(*list(orig_resnet.children())[:-2])
        
        self.fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
        
    def forward(self, x, pool=True):
        x = self.features(x)
        
        x = self.fc(x)
        if not pool:
            return x
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), x.size(1))      
        return x
        
    def forward_multiframe(self, x, pool=True):
        
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)
        
        x = self.features(x)
        
        x = self.fc(x)
        
        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        x = x.permute(0, 2, 1, 3, 4)  
        
        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        
        x = x.view(B, C)
        
        return x
        
class ClipVisual(nn.Module):
    def __init__(self, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisual, self).__init__()
        from functools import partial
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        orig_resnet = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = orig_resnet.state_dict()
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            new_k = k.replace('visual.', '')
            if new_k in model_dict:
                updated_pretrained_dict[new_k] = v
                
        model_dict.update(updated_pretrained_dict)
        orig_resnet.load_state_dict(model_dict)

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
        
        self.features = nn.Sequential(*list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        
        return x
        
    def forward_center_frame(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.squeeze(2)
        
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        
        x = x.permute(0, 2, 1, 3, 4)  

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)

        x = x.view(B, C)
        
        return x

class ClipVisualVitOrig(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualVitOrig, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        
        state_dict = torch.load(pretrained_path)
        
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        orig_resnet = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)
            
        pretrained_dict = state_dict
        model_dict = orig_resnet.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if 'visual.' not in k:
                continue
        
            new_k = k.replace('visual.', '')
            if new_k in model_dict:
                updated_pretrained_dict[new_k] = v
                
        model_dict.update(updated_pretrained_dict)
        orig_resnet.load_state_dict(model_dict)

        '''if 'full-mean' in self.arch:
            self.features = nn.Sequential(*list(orig_resnet.children())[:])
        elif 'region' in self.arch:
        
            if '-top' in self.arch:
                self.features = nn.Sequential(*list(orig_resnet.children())[:])
            else:
                self.features = nn.Sequential(*list(orig_resnet.children())[:-2])'''
                
        self.features = orig_resnet
            
        # Freeze clip
        for k, v in self.features.named_parameters():
          v.requires_grad = False
          
        if '-conv' in self.arch:
            self.fc = nn.Conv2d(512, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif '-fc' in self.arch:
            self.fc = nn.Linear(512, fc_dim)
            
    def forward_multiframe(self, x, pool=True):
        
        (B, C, T, H, W) = x.size()
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)
        
        x = self.features(x)

        if 'full-mean-center' in self.arch:
            x = x[:, 0, :]
            x = x.view(B, T, -1)
            x = x[:, 0, :]

        elif 'full-mean' in self.arch:
            x = x[:, 0, :]
            
            x = x.view(B, T, -1)
            x = x.mean(1)
            
            if '-fc' in self.arch:
                x = self.fc(x)
            
        elif 'region' in self.arch:
            x = x[:, 1:, :]
            x = x.permute(0, 2, 1)
            x = x.view(x.size(0), x.size(1), 7, 7)
            
            if '-conv' in self.arch:
                x = self.fc(x)
                
            elif '-fc' in self.arch:            
                x = x.permute(0, 2, 3, 1)
                x = self.fc(x)
                x = x.permute(0, 3, 1, 2)
                
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            
            if 'maxpool' in self.arch:
                x = F.adaptive_max_pool3d(x, 1)
            elif 'meanpool' in self.arch:
                x = F.adaptive_avg_pool3d(x, 1)
                
            x = x.view(B, C)
        
        return x
        
class ClipVisualVitJoint(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=512, vis_dim=512):
        super(ClipVisualVitJoint, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        
        state_dict = torch.load(pretrained_path)
        
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = DebugVisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        
        if 'post-fc' in self.arch:
            self.text_fc = nn.Linear(text_dim, fc_dim)
            if '-linear' in self.arch:
                self.vis_fc = nn.Linear(vis_dim, fc_dim)
            else:
                self.vis_fc = nn.Conv2d(vis_dim, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif 'pre-fc' in self.arch:
            self.vis_fc = nn.Linear(vis_dim, fc_dim)
        self.softmax = nn.Softmax(dim=-1)
            
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
          
    def compute_agg_visual(self, vis, text):
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = vis / vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)
        scores = self.softmax(scores)
        agg_vis = vis * scores.unsqueeze(-1)
        agg_vis = agg_vis.sum(1)
        return agg_vis, scores
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)

        x = self.visual(x, self.arch)
        
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        
        if 'post-fc' in self.arch:
        
            if '-linear' in self.arch:
                x = x.permute(0, 2, 3, 1)
                x = self.vis_fc(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
                x = self.vis_fc(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, (B, -1, x.size(-1)))
                
            text = self.text_fc(text)
        else:
        
            # normalized features
            x = x / x.norm(dim=-1, keepdim=True)
            text = text / text.norm(dim=-1, keepdim=True)
            x = torch.reshape(x, (B, -1, x.size(-1)))

        agg_x, scores = self.compute_agg_visual(x, text)
        scores = scores.view(B, T, spatial_dim, -1)
        
        
        '''(B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = x.view(B*T, C, H, W)
        
        x = self.features(x)
        
        if 'post-fc' in self.arch:
        
            x = x[:, 1:, :]
            x = x.view(x.size(0), x.size(-1), 7, 7)
            (_, C, H, W) = x.size()
        
            if '-linear' in self.arch:
                x = x.permute(0, 2, 3, 1)
                x = self.vis_fc(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = self.vis_fc(x)
            text = self.text_fc(text)
            
            # compute text-aggregated visual features
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            
            agg_x, scores = self.compute_agg_visual(x, text)
        elif 'pre-fc' in self.arch:
        
            x = x[:, 1:, :]
            x = x.view(x.size(0), x.size(-1), 7, 7)
            (_, C, H, W) = x.size()
            
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            
            agg_x, scores = self.compute_agg_visual(x, text)
            agg_x = self.vis_fc(agg_x)    
            
        elif 'orig-dim' in self.arch:
        
            x = x[:, 1:, :]
            x = x.view(x.size(0), x.size(-1), 7, 7)
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            agg_x, scores = self.compute_agg_visual(x, text)'''
        
        return agg_x, scores
        
class ClipVisualVitDebug(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualVitDebug, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        
        state_dict = torch.load(pretrained_path)
        
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = DebugVisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
            
        if 'finetune' not in self.arch:            
            # Freeze clip
            for k, v in self.named_parameters():
                v.requires_grad = False
            
    def forward_multiframe(self, x, pool=True):
    
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x, self.arch)
        
        if 'full-mean' in self.arch:
        
            x = x.view(B, T, -1)
            if 'center' in self.arch:
                x = x[:, 1, :]
            else:
                x = x.mean(1)
        elif 'region' in self.arch:        
            x = x[:, 1:, :]
            (_, _, C) = x.size()
            spatial_dim = int(x.size(1) ** 0.5)
            
            x = x.view(B, T, spatial_dim, spatial_dim, -1)
            x = x.permute(0, 4, 1, 2, 3)
            
            if 'maxpool' in self.arch:
                x = F.adaptive_max_pool3d(x, 1)
            elif 'meanpool' in self.arch:
                x = F.adaptive_avg_pool3d(x, 1)
                
            x = x.view(B, C)
        
        return x
        
class ImageNetVit(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ImageNetVit, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        
        self.visual = ViT('B_32', pretrained=False)
        self.visual.load_state_dict(torch.load(pretrained_path))
            
        # Freeze clip
        for k, v in self.named_parameters():
          v.requires_grad = False
            
    def forward_multiframe(self, x, pool=True):
    
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x)
        
        if 'center' in self.arch:
            x = x[:, 0, :]
            x = x.view(B, T, -1)
            x = x[:, 1, :]
        elif 'full-mean' in self.arch:
            x = x[:, 0, :]
            x = x.view(B, T, -1)
            x = x.mean(1)
        elif 'region' in self.arch:
            (_, _, C) = x.size()
        
            x = x[:, 1:, :]
            spatial_dim = int(x.size(1) ** 0.5)
            
            x = x.view(B, T, spatial_dim, spatial_dim, -1)
            x = x.permute(0, 4, 1, 2, 3)
            
            if 'maxpool' in self.arch:
                x = F.adaptive_max_pool3d(x, 1)
            elif 'meanpool' in self.arch:
                x = F.adaptive_avg_pool3d(x, 1)
                
            x = x.view(B, C)
        
        return x
        
class ClipVisualResDistill(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualResDistill, self).__init__()
        from functools import partial

        self.arch = arch
        self.pool_type = pool_type
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = DebugModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
           
        if 'mlp-conv' in self.arch:
            self.fc = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=conv_size, padding=conv_size//2), 
                                    nn.ReLU(), 
                                    nn.Conv2d(512, fc_dim, kernel_size=conv_size, padding=conv_size//2))
        elif '-fc' in self.arch:
            self.fc = nn.Linear(1024, fc_dim)
        else:
            self.fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
            
        # Freeze clip
        for k, v in self.visual.named_parameters():
          v.requires_grad = False
            
    def forward_multiframe(self, x, pool=True):
    
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x, self.arch)
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        
        if '-fc' in self.arch:
            x = self.fc(x)
            (_, _, C) = x.size()
            x = x.permute(0, 2, 1)
            x = x.view(B, T, -1, spatial_dim, spatial_dim)
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = x.permute(0, 2, 1)
            x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
        
            x = self.fc(x)
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
            
        x = x.view(B, C)
        
        return x
        
class ClipVisualVitDistill(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ClipVisualVitDistill, self).__init__()
        from functools import partial

        self.arch = arch
        self.pool_type = pool_type
        
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_vitb32.pth"
        
        state_dict = torch.load(pretrained_path)
        
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = DebugVisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)
           
        if 'mlp-conv' in self.arch:
            self.fc = nn.Sequential(nn.Conv2d(512, 256, kernel_size=conv_size, padding=conv_size//2), 
                                    nn.ReLU(), 
                                    nn.Conv2d(256, fc_dim, kernel_size=conv_size, padding=conv_size//2))   
        elif '-fc' in self.arch:
            self.fc = nn.Linear(512, fc_dim)
        else:
            self.fc = nn.Conv2d(512, fc_dim, kernel_size=conv_size, padding=conv_size//2)
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
        
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
            
        # Freeze clip
        for k, v in self.visual.named_parameters():
          v.requires_grad = False
          
    def forward_multiframe(self, x, pool=True):
    
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x, self.arch)
        x = x[:, 1:, :]
        (_, _, C) = x.size()
        spatial_dim = int(x.size(1) ** 0.5)
        
        if '-fc' in self.arch:
            x = self.fc(x)
            (_, _, C) = x.size()
            x = x.permute(0, 2, 1)
            x = x.view(B, T, -1, spatial_dim, spatial_dim)
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = x.permute(0, 2, 1)
            x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
            x = self.fc(x)
        
            (_, C, H, W) = x.size()
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
            
        x = x.view(B, C)
        
        return x

class ImageNetVitJoint(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ImageNetVitJoint, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        
        self.visual = ViT('B_32', pretrained=False)
        self.visual.load_state_dict(torch.load(pretrained_path))
        
        if 'vitb32-text' in self.arch:
            text_dim = 512
        else:
            text_dim = 1024
        
        self.fc = nn.Conv2d(768, text_dim, kernel_size=conv_size, padding=conv_size//2)
        self.softmax = nn.Softmax(dim=-1)
            
        # Freeze clip
        for k, v in self.visual.named_parameters():
          v.requires_grad = False
          
    def compute_agg_visual(self, vis, text):
        scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)        
        scores = self.softmax(scores)
        agg_vis = vis * scores.unsqueeze(-1)
        agg_vis = agg_vis.sum(1)
        return agg_vis, scores
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x)
        
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
        
        x = self.fc(x)
        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, -1, C)

        agg_x, scores = self.compute_agg_visual(x, text)
        scores = scores.view(B, T, spatial_dim, -1)
        
        return agg_x, scores
        
class ImageNetVitDistill(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ImageNetVitDistill, self).__init__()
        from functools import partial

        self.arch = arch
        
        pretrained_path = "/research/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/object-detection/models/Sound-of-Pixels/pretrained_model_weights/imagenet_vitb32_21k.pth"
        
        self.visual = ViT('B_32', pretrained=False)
        self.visual.load_state_dict(torch.load(pretrained_path))
        
        self.fc = nn.Conv2d(768, fc_dim, kernel_size=conv_size, padding=conv_size//2)
            
        # Freeze clip
        for k, v in self.visual.named_parameters():
          v.requires_grad = False
            
    def forward_multiframe(self, x, pool=True):
    
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        x = self.visual(x)
        
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        x = x.permute(0, 2, 1)
        
        x = x.view(x.size(0), x.size(1), spatial_dim, spatial_dim)
        x = self.fc(x)
        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = F.adaptive_max_pool3d(x, 1)
        x = x.view(B, C)
        
        return x
        
class ClipVisualResDetr(nn.Module):
    def __init__(self, args, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dim_feedforward=2048, num_layers=1, num_attn_heads=2, dropout=0.1):
        super(ClipVisualResDetr, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        self.softmax = nn.Softmax(dim=-1)
        
        # positional embeddings for audio spectrogram features
        if 'upsample3' in args.arch_sound:
            self.audio_positional_embedding = nn.Parameter(torch.randn(256, fc_dim) / fc_dim ** 0.5)
        else:
            self.audio_positional_embedding = nn.Parameter(torch.randn(64, fc_dim) / fc_dim ** 0.5)
            
        # create sound source query embeddings
        num_queries = 1 # hardcoded to 1 for now
        self.query_embed = nn.Embedding(num_queries, fc_dim)
            
        # creates multimodal transformer encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(fc_dim, num_attn_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(fc_dim, num_attn_heads, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # create vocoder
        self.synthesizer = nn.Linear(fc_dim, 256*256)
        
        self.selected_out = OrderedDict()
        self.intermediate_results = []
            
        self.intermediate_results.append(getattr(self.transformer_encoder._modules['layers']._modules[str(num_layers-1)], 'self_attn').register_forward_hook(self.forward_hook('self_attn')))
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
        elif 'finetune' in self.arch and 'traintext' in self.arch:
            for k, v in self.visual.named_parameters():
                if 'attnpool.c_proj' not in k and 'attnpool.v_proj' not in k:
                    v.requires_grad = False
                
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
            
    def forward_multiframe(self, x, text, audio, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)

        x = self.visual(x, self.arch)
        
        x = x[:, 1:, :]
        vis_spatial_dim = int(x.size(1) ** 0.5)
        
        aud_spatial_dim = audio.size(-1)
        audio = audio.view(B, audio.size(1), -1)
        audio = audio.permute(0, 2, 1).contiguous()
        
        num_vis_regions = x.size(1)
        num_aud_regions = audio.size(1)
        
        # concatenate text, visual and audio features into 1D sequence
        audio = audio + self.audio_positional_embedding[None, :, :]
        x = torch.reshape(x, (B, -1, x.size(-1)))
        
        if 'normalize' in self.arch:
            text = text / text.norm(dim=-1, keepdim=True)
            x = x / x.norm(dim=-1, keepdim=True)
        
        input_seq = torch.cat([text.unsqueeze(1), x, audio], 1)
        
        #if 'normalize' in self.arch:
        #    input_seq = input_seq / input_seq.norm(dim=-1, keepdim=True)
        
        hidden_states = self.transformer_encoder(input_seq.permute(1, 0, 2))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        decoder_output = self.transformer_decoder(query_embed.permute(1, 0, 2), hidden_states)
        decoder_output = decoder_output.permute(1, 0, 2)
        decoder_output = self.synthesizer(decoder_output)
        decoder_output = decoder_output.view(decoder_output.size(0), decoder_output.size(1), int(decoder_output.size(-1) ** 0.5), -1)
        decoder_output = torch.sigmoid(decoder_output)
        
        attn_head_output = self.selected_out['self_attn']
        visual_attn_weights = attn_head_output[1]
        visual_attn_weights = visual_attn_weights[:, 0, 1:1+(num_vis_regions*3)]
        visual_attn_weights = visual_attn_weights.view(B, T, vis_spatial_dim, vis_spatial_dim)
            
        self.selected_out.clear()
        
        return decoder_output, visual_attn_weights
        
class ClipJointAttn(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(ClipJointAttn, self).__init__()
        from functools import partial
        
        self.arch = arch
        self.arch = 'res50-textproj-framewise'
        
        if 'res50' in self.arch:
            pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
            state_dict = torch.load(pretrained_path)
            
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
    
            embed_dim = state_dict["text_projection"].shape[1]
            context_length = state_dict["positional_embedding"].shape[0]
            vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
            vision_heads = vision_width * 32 // 64
        
            self.visual = DebugModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
            self.context_length = context_length
        
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )

            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
            
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                               
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)

        self.softmax = nn.Softmax(dim=-1)
                
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, vis, text, num_frames):
        spatial_dim = int(vis.size(-2) ** 0.5)
    
        norm_vis = vis / vis.norm(dim=-1, keepdim=True)
        norm_text = text / text.norm(dim=-1, keepdim=True)
        norm_vis = norm_vis.view(norm_vis.size(0), -1, norm_vis.size(-1))
        
        scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        
        if 'spatiotemporal' in self.arch:
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), num_frames, spatial_dim, -1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), num_frames, spatial_dim, -1)
        
        return scores
            
    def forward_multiframe(self, text, frames):
        (B, T, C, H, W) = frames.size()
        
        frames = frames.view(B*T, C, H, W)
        
        text_features = self.encode_text(text)
        frames_features = self.visual(frames, self.arch)
        
        if 'clsattn' in self.arch:
            scores = frames_features
            spatial_dim = int(scores.size(1) ** 0.5)
            scores = scores.view(B, T, -1)
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), scores.size(1), spatial_dim, -1)
            return scores
        elif 'contextnocls' in self.arch:
            region_features = frames_features
        else:
            cls_features = frames_features[:, 0, :]
            region_features = frames_features[:, 1:, :]    
        
        spatial_dim = int(region_features.size(1) ** 0.5)
        region_features = region_features.view(B, T, -1, region_features.size(-1)).contiguous()
        
        scores = self.compute_agg_visual(region_features, text_features, T)
    
        return scores

class ClipVisualResDistillAttn(nn.Module):
    def __init__(self, args, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(ClipVisualResDistillAttn, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        self.logit_scale = nn.Parameter(state_dict['logit_scale']) # use this for new frame ablation jobs !!!!!
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        
        self.scale = 'scale' in self.arch
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
            self.logit_scale.requires_grad = False
        elif 'finetune' in self.arch and 'traintext' in self.arch:
            for k, v in self.visual.named_parameters():            
                if 'attnpool.c_proj' not in k and 'attnpool.v_proj' not in k:
                    v.requires_grad = False
                    
        self.logit_scale.requires_grad = False
                    
        for module in self.visual.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, vis, text, num_frames):
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = vis / vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(vis, text.unsqueeze(-1)).squeeze(-1)
        
        # Test runs with CLIP scaling factor
        if self.scale:
            logit_scale = self.logit_scale.exp()   
            scores = logit_scale * scores
            
        if 'spatiotemporal' in self.arch:
            scores = F.softmax(scores, dim=-1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = F.softmax(scores, dim=-1)            
            scores = scores.view(scores.size(0), -1)
        
        return scores # scores = (B, 147)
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        if 'combinedvis' in self.arch:
            context_x, no_context_x = self.visual(x, self.arch)
            context_x = context_x[:, 1:, :]
            no_context_x = no_context_x[:, 1:, :]
            
            spatial_dim = int(context_x.size(1) ** 0.5)
            no_context_x = torch.reshape(no_context_x, (B, -1, no_context_x.size(-1)))
            scores = self.compute_agg_visual(no_context_x, text, T)
            context_x = context_x.view(B, T, spatial_dim, spatial_dim, -1)
            
            return context_x, scores
        
        x = self.visual(x, self.arch)
        x = x[:, 1:, :]
        spatial_dim = int(x.size(1) ** 0.5)
        
        x = torch.reshape(x, (B, -1, x.size(-1)))
        scores = self.compute_agg_visual(x, text, T)
        x = x.view(x.size(0), T, spatial_dim, spatial_dim, -1)
        
        return x, scores
        
class LatentJointAttn(nn.Module):
    def __init__(self, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(LatentJointAttn, self).__init__()
        from functools import partial
        
        self.arch = arch
        
        if 'res50' in self.arch:
            pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
            state_dict = torch.load(pretrained_path)
            
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
    
            embed_dim = state_dict["text_projection"].shape[1]
            context_length = state_dict["positional_embedding"].shape[0]
            vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
            vision_heads = vision_width * 32 // 64
        
            self.visual = DebugModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
            self.context_length = context_length
        
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )

            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()
            
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                               
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        self.softmax = nn.Softmax(dim=-1)
                
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, vis, text, num_frames):
        spatial_dim = int(vis.size(-2) ** 0.5)
    
        norm_vis = vis / vis.norm(dim=-1, keepdim=True)
        norm_text = text / text.norm(dim=-1, keepdim=True)
        norm_vis = norm_vis.view(norm_vis.size(0), -1, norm_vis.size(-1))
        scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        
        if 'spatiotemporal' in self.arch:
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), num_frames, spatial_dim, -1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), num_frames, spatial_dim, -1)
        
        return scores
            
    def forward_multiframe(self, text, frames):
        (B, T, C, H, W) = frames.size()
        
        frames = frames.view(B*T, C, H, W)
        
        #text_features = self.encode_text(text)
        frames_features = self.visual(frames, self.arch)
        
        if 'clsattn' in self.arch:
            scores = frames_features
            spatial_dim = int(scores.size(1) ** 0.5)
            scores = scores.view(B, T, -1)
            scores = self.softmax(scores)
            scores = scores.view(scores.size(0), scores.size(1), spatial_dim, -1)
            return scores
        elif 'contextnocls' in self.arch:
            region_features = frames_features
        else:
            cls_features = frames_features[:, 0, :]
            region_features = frames_features[:, 1:, :]    
        
        spatial_dim = int(region_features.size(1) ** 0.5)
        region_features = region_features.view(B, T, -1, region_features.size(-1)).contiguous()
        
        scores = self.compute_agg_visual(region_features, text, T)
    
        return scores
        
class ClipVisualCombinedJoint(nn.Module):
    def __init__(self, args, arch, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, text_dim=1024, vis_dim=1024, dropout=0.1):
        super(ClipVisualCombinedJoint, self).__init__()
        from functools import partial
        pretrained_path = "/research/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs5/mnt/data/reuben/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        #self.logit_scale = state_dict['logit_scale']
        self.logit_scale = nn.Parameter(state_dict['logit_scale']) # use this for new frame ablation jobs !!!!!
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            
        pretrained_dict = state_dict
        model_dict = self.state_dict()

        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v

        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        self.arch = arch
        
        self.scale = 'scale' in self.arch
        
        if 'post-fc' in self.arch:
            self.text_fc = nn.Linear(text_dim, fc_dim)
            if '-linear' in self.arch:
                self.vis_fc = nn.Linear(vis_dim, fc_dim)
            else:
                self.vis_fc = nn.Conv2d(1024, fc_dim, kernel_size=conv_size, padding=conv_size//2)
        elif 'pre-fc' in self.arch:
            self.vis_fc = nn.Linear(vis_dim, fc_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        if 'attn' in self.arch:
            self.multihead_attn = nn.MultiheadAttention(1024, args.num_attention_heads, batch_first=True)
        elif 'trans' in self.arch:
            encoder_layer = nn.TransformerEncoderLayer(1024, args.num_attention_heads, args.ffn_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)
            
            self.selected_out = OrderedDict()
            self.intermediate_results = []
            
            self.intermediate_results.append(getattr(self.transformer_encoder._modules['layers']._modules[str(args.num_transformer_layers-1)], 'self_attn').register_forward_hook(self.forward_hook('self_attn')))
        
        # Freeze text encoder
        if 'finetune' not in self.arch:
            for k, v in self.visual.named_parameters():
                v.requires_grad = False
            self.logit_scale.requires_grad = False
        elif 'finetune' in self.arch and 'traintext' in self.arch:
            for k, v in self.visual.named_parameters():            
                if 'attnpool.c_proj' not in k and 'attnpool.v_proj' not in k:
                    v.requires_grad = False
                    
        self.logit_scale.requires_grad = False
                    
        for module in self.visual.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
          
    def compute_agg_visual(self, context_vis, no_context_vis, text, num_frames):
    
        if'-normalize' in self.arch:
            # normalized features
            norm_vis = no_context_vis / no_context_vis.norm(dim=-1, keepdim=True)
            norm_text = text / text.norm(dim=-1, keepdim=True)
            scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.bmm(no_context_vis, text.unsqueeze(-1)).squeeze(-1)
            
        # Test runs with CLIP scaling factor
        if self.scale:
            logit_scale = self.logit_scale.exp()    
            scores = logit_scale * scores
    
        if 'spatiotemporal' in self.arch:
            scores = self.softmax(scores)
            if 'normalize-sum' in self.arch:
                agg_vis = norm_vis * scores.unsqueeze(-1)
            else:
                agg_vis = vis * scores.unsqueeze(-1)
            agg_vis = agg_vis.sum(1)
        elif 'framewise' in self.arch:
            scores = scores.view(scores.size(0), num_frames, -1)
            scores = self.softmax(scores)
            context_vis = context_vis.view(context_vis.size(0), num_frames, -1, context_vis.size(-1))
            agg_vis = context_vis * scores.unsqueeze(-1)
            agg_vis = agg_vis.sum(-2)
            
            if 'maxpool' in self.arch:
                agg_vis = torch.max(agg_vis, dim=1)[0]
            else:
                agg_vis = agg_vis.mean(1)
        
        return agg_vis, scores # scores = (B, 147)
            
    def forward_multiframe(self, x, text, pool=True):
        text = text.float()
        (B, T, C, H, W) = x.size()
        x = x.view(B*T, C, H, W)
        
        context_x, no_context_x = self.visual(x, self.arch)
        context_x = context_x[:, 1:, :]
        no_context_x = no_context_x[:, 1:, :]
        
        spatial_dim = int(context_x.size(1) ** 0.5)
        
        context_x = torch.reshape(context_x, (B, -1, context_x.size(-1)))
        no_context_x = torch.reshape(no_context_x, (B, -1, no_context_x.size(-1)))
        
        agg_x, scores = self.compute_agg_visual(context_x, no_context_x, text, T)
        scores = scores.view(B, T, spatial_dim, -1)
        
        return agg_x, scores