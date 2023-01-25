import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
class FTLatentDistillAudioVisual7layerUNet(nn.Module):
    def __init__(self, args, ngf=64, input_nc=1, output_nc=1):
        super(FTLatentDistillAudioVisual7layerUNet, self).__init__()
        
        self.args = args
        self.fc_dim = args.num_channels
        self.audio_arch = args.arch_sound
        self.visual_arch = args.arch_frame
        self.num_frames = args.num_frames
        
        if self.fc_dim == 512:
            self.dim_upsample_factor = 1
        elif self.fc_dim == 1024:
            self.dim_upsample_factor = 2
            
        latent_concepts = torch.from_numpy(np.load(args.latent_concept_path))
        self.latent_concepts = nn.Parameter(latent_concepts, requires_grad=True)

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer6 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer7 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
        if 'textclass' in self.visual_arch or 'combined' in self.visual_arch:
            self.ce_loss = nn.CrossEntropyLoss()
            
        if 'mlp' in self.audio_arch:
            self.audio_mlp = nn.Sequential(nn.Linear(self.fc_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.fc_dim))
        
    def compute_audio_attn(self, audio_conv7feature, visual_feats, text_cat_idx):
        B, T, S, S, D = visual_feats.shape
        
        visual_feats = visual_feats.view(B, -1, D)
        
        norm_vis = visual_feats / visual_feats.norm(dim=-1, keepdim=True)
        norm_aud = audio_conv7feature / audio_conv7feature.norm(dim=-1, keepdim=True)
        
        scores = torch.bmm(norm_vis, norm_aud.unsqueeze(-1))
        
        scores = scores * 100.
        
        if 'framewise' in self.visual_arch:
            scores = scores.view(B, T, -1)
        else:
            scores = scores.view(B, -1)
            
        scores = F.log_softmax(scores, dim=-1)
        
        text_feats = self.latent_concepts[text_cat_idx]        
        norm_text = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_scores = torch.bmm(norm_vis, norm_text.unsqueeze(-1)).squeeze(-1)
        text_scores = text_scores * 100.
        text_scores = text_scores.view(text_scores.size(0), self.num_frames, -1)
        text_scores = F.softmax(text_scores, dim=-1)
        text_scores = text_scores.view(text_scores.size(0), -1)
        
        return scores, text_scores
        
    def encode_audio(self, audio):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        return audio_conv7feature.squeeze()
        
    def compute_audio_text_sim(self, audio_feat, combined_cat_idx):    
        norm_text = self.latent_concepts / self.latent_concepts.norm(dim=-1, keepdim=True)
        norm_text = norm_text.to(audio_feat.device)
        norm_aud = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
        scores = torch.matmul(norm_aud, norm_text.t())
        
        scores = scores * 100
        
        text_class_loss = self.ce_loss(scores, combined_cat_idx)
        return text_class_loss
        
        
    def compute_text_classification(self, audio_mag_mix, pred_masks, first_cat_idx, second_cat_idx):
        combined_cat_idx = torch.cat((first_cat_idx, second_cat_idx), dim=0)
        combined_pred_mask = torch.cat((pred_masks[0], pred_masks[1]), dim=0)
        combined_mag_mix = audio_mag_mix.tile((len(pred_masks), 1, 1, 1))
        
        combined_pred_mag = combined_mag_mix * combined_pred_mask
        combined_aud_feature = self.encode_audio(combined_pred_mag)
        
        if 'mlp' in self.audio_arch:
            combined_aud_feature = self.audio_mlp(combined_aud_feature)
        
        text_class_loss = self.compute_audio_text_sim(combined_aud_feature, combined_cat_idx)
    
        return text_class_loss
        
        
    def encode_spec(self, x, visual_feats, text_cat_idx):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        audio_conv7feature = audio_conv7feature.squeeze()
        
        if 'mlp' in self.audio_arch:
            audio_conv7feature = self.audio_mlp(audio_conv7feature)
        
        scores = self.compute_audio_attn(audio_conv7feature, visual_feats, text_cat_idx)
        return scores
        
        
    def upsample_conv(self, all_down_conv_features, feat):
        audio_upconv1feature = self.audionet_upconvlayer1(feat)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5]), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4]), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3]), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2]), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1]), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0]), dim=1))
        return mask_prediction
        
    def forward_text(self, x, text):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = text[0]
        second_combined = text[1]
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction

    def forward(self, x, frame_feats):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = frame_feats[0].permute(0, 4, 1, 2, 3)
        second_combined = frame_feats[1].permute(0, 4, 1, 2, 3)
        
        if 'maxpool' in self.visual_arch:
            first_combined = F.adaptive_max_pool3d(first_combined, 1)
            second_combined = F.adaptive_max_pool3d(second_combined, 1)
        else:
            first_combined = F.adaptive_avg_pool3d(first_combined, 1)
            second_combined = F.adaptive_avg_pool3d(second_combined, 1)
            
        first_combined = first_combined.squeeze()
        second_combined = second_combined.squeeze()
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction
        
class BimodalDistillAudioVisual7layerUNet(nn.Module):
    def __init__(self, args, ngf=64, input_nc=1, output_nc=1):
        super(BimodalDistillAudioVisual7layerUNet, self).__init__()
        
        self.args = args
        self.fc_dim = args.num_channels
        self.audio_arch = args.arch_sound
        self.visual_arch = args.arch_frame
        self.use_latent_concepts = args.use_latent_concepts
        self.num_regions = 49
        
        if self.fc_dim == 512:
            self.dim_upsample_factor = 1
        elif self.fc_dim == 1024:
            self.dim_upsample_factor = 2
       
        if self.use_latent_concepts:
            precomputed_text_features = np.load(args.latent_concept_path)
        else:
            precomputed_text_features = np.load('./precomputed_features/solos_clip_rn50_text_category_features.npy')
        self.precomputed_text_features = torch.from_numpy(precomputed_text_features).float()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer6 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer7 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
        if 'textclass' in self.visual_arch or 'combined' in self.visual_arch:
            self.ce_loss = nn.CrossEntropyLoss()
            
        if 'mlp' in self.audio_arch:
            self.audio_mlp = nn.Sequential(nn.Linear(self.fc_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.fc_dim))
        
    def encode_audio(self, audio):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        return audio_conv7feature.squeeze()
        
    def compute_audio_text_sim(self, audio_feat, combined_cat_idx):
    
        norm_text = self.precomputed_text_features / self.precomputed_text_features.norm(dim=-1, keepdim=True)
        norm_text = norm_text.to(audio_feat.device)
        norm_aud = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
        scores = torch.matmul(norm_aud, norm_text.t())
        
        scores = scores * 100
        
        text_class_loss = self.ce_loss(scores, combined_cat_idx)
        return text_class_loss
        
    def compute_classification_loss(self, combined_pred_audio_feats, combined_cat_idx):
        text_class_loss = self.compute_audio_text_sim(combined_pred_audio_feats, combined_cat_idx)
        return text_class_loss
        
        
    def compute_text_classification(self, audio_mag_mix, pred_masks, first_cat_idx, second_cat_idx):
        combined_cat_idx = torch.cat((first_cat_idx, second_cat_idx), dim=0)
        combined_pred_mask = torch.cat((pred_masks[0], pred_masks[1]), dim=0)
        combined_mag_mix = audio_mag_mix.tile((len(pred_masks), 1, 1, 1))
        
        combined_pred_mag = combined_mag_mix * combined_pred_mask
        combined_aud_feature = self.encode_audio(combined_pred_mag)
        
        if 'mlp' in self.audio_arch:
            combined_aud_feature = self.audio_mlp(combined_aud_feature)
        
        text_class_loss = self.compute_audio_text_sim(combined_aud_feature, combined_cat_idx)
    
        return text_class_loss
        
    def encode_audio_spec(self, audio_mag_mix, pred_masks):
        pred_spec = audio_mag_mix * pred_masks
        pred_audio_feature = self.encode_audio(pred_spec)
    
        return pred_audio_feature
        
    def compute_audio_attn(self, audio_conv7feature, visual_feats, use_softmax):
        B, T, S, S, D = visual_feats.shape
        
        visual_feats = visual_feats.view(B, -1, D)
        
        norm_vis = visual_feats / visual_feats.norm(dim=-1, keepdim=True)
        norm_aud = audio_conv7feature / audio_conv7feature.norm(dim=-1, keepdim=True)
        
        scores = torch.bmm(norm_vis, norm_aud.unsqueeze(-1))
        
        scores = scores * 100
        
        if 'framewise' in self.visual_arch:
            scores = scores.view(B, T, -1)
        else:
            scores = scores.view(B, -1)
            
        if use_softmax:
            scores = F.softmax(scores, dim=-1)
        else:
            scores = F.log_softmax(scores, dim=-1)
            
        return scores
        
    def compute_audio_to_video_attn(self, pred_audio_feats, visual_feats, use_softmax=False):
        scores = self.compute_audio_attn(pred_audio_feats, visual_feats, use_softmax)
        return scores
        
        
    def encode_spec(self, x, visual_feats, use_softmax=False):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        audio_conv7feature = audio_conv7feature.squeeze()
        
        if 'mlp' in self.audio_arch:
            audio_conv7feature = self.audio_mlp(audio_conv7feature)
        
        scores = self.compute_audio_attn(audio_conv7feature, visual_feats, use_softmax)
        return scores
        
        
    def upsample_conv(self, all_down_conv_features, feat, num_regions=None):
    
        if 'regionmask' in self.visual_arch:
            audio_upconv1feature = self.audionet_upconvlayer1(feat)
            audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[5].size(-3), all_down_conv_features[5].size(-2), all_down_conv_features[5].size(-1))), dim=1))
            audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[4].size(-3), all_down_conv_features[4].size(-2), all_down_conv_features[4].size(-1))), dim=1))
            audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[3].size(-3), all_down_conv_features[3].size(-2), all_down_conv_features[3].size(-1))), dim=1))
            audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[2].size(-3), all_down_conv_features[2].size(-2), all_down_conv_features[2].size(-1))), dim=1))
            audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[1].size(-3), all_down_conv_features[1].size(-2), all_down_conv_features[1].size(-1))), dim=1))
            mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[0].size(-3), all_down_conv_features[0].size(-2), all_down_conv_features[0].size(-1))), dim=1))
            mask_prediction = mask_prediction.view(-1, num_regions, mask_prediction.size(-3), mask_prediction.size(-2), mask_prediction.size(-1))
            
            mask_prediction = mask_prediction.sum(1)
            
            #mask_prediction = torch.max(mask_prediction, dim=1)[0]
            
            #print('mask_prediction: ', mask_prediction)
            #print('mask_prediction: ', mask_prediction.shape)
            #sys.exit()
            
            return mask_prediction
    
        audio_upconv1feature = self.audionet_upconvlayer1(feat)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5]), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4]), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3]), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2]), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1]), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0]), dim=1))
        return mask_prediction
        
    def forward_text(self, x, text):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = text[0]
        second_combined = text[1]
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction

    def forward(self, x, frame_feats, text):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        
        first_combined = frame_feats[0]
        second_combined = frame_feats[1]
        
        if 'regionmask' in self.visual_arch:
            first_combined = first_combined.view(first_combined.size(0), -1, first_combined.size(-1))
            second_combined = second_combined.view(second_combined.size(0), -1, second_combined.size(-1))
            
            audio_conv7feature = audio_conv7feature.unsqueeze(1).repeat(1, first_combined.size(1), 1, 1, 1)
            first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
            second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
            num_regions = first_combined.size(1)
            first_combined = torch.cat((first_combined, audio_conv7feature), dim=2)
            second_combined = torch.cat((second_combined, audio_conv7feature), dim=2)
            
            first_combined = first_combined.view(-1, first_combined.size(-3), first_combined.size(-2), first_combined.size(-1))
            second_combined = second_combined.view(-1, second_combined.size(-3), second_combined.size(-2), second_combined.size(-1))
            
            first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined, num_regions)
            second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined, num_regions)
           
            return first_mask_prediction, second_mask_prediction
            
        first_combined = first_combined.permute(0, 4, 1, 2, 3)
        second_combined = second_combined.permute(0, 4, 1, 2, 3)
        
        if 'maxpool' in self.visual_arch:
            first_combined = F.adaptive_max_pool3d(first_combined, 1)
            second_combined = F.adaptive_max_pool3d(second_combined, 1)
        else:
            first_combined = F.adaptive_avg_pool3d(first_combined, 1)
            second_combined = F.adaptive_avg_pool3d(second_combined, 1)
            
        first_combined = first_combined.squeeze()
        second_combined = second_combined.squeeze()
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        # compute masks for latent captions
        first_text_combined = text[0]
        second_text_combined = text[1]
        
        first_text_combined = first_text_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_text_combined = second_text_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        
        first_text_combined = torch.cat((first_text_combined, audio_conv7feature), dim=1)
        second_text_combined = torch.cat((second_text_combined, audio_conv7feature), dim=1)

        first_text_mask_prediction = self.upsample_conv(all_down_conv_features, first_text_combined)
        second_text_mask_prediction = self.upsample_conv(all_down_conv_features, second_text_combined)
        
        return first_mask_prediction, second_mask_prediction, first_text_mask_prediction, second_text_mask_prediction
        
class DistillAudioVisual7layerUNet(nn.Module):
    def __init__(self, args, ngf=64, input_nc=1, output_nc=1):
        super(DistillAudioVisual7layerUNet, self).__init__()
        
        self.args = args
        self.fc_dim = args.num_channels
        self.audio_arch = args.arch_sound
        self.visual_arch = args.arch_frame
        self.use_latent_concepts = args.use_latent_concepts
        self.num_regions = 49
        
        if self.fc_dim == 512:
            self.dim_upsample_factor = 1
        elif self.fc_dim == 1024:
            self.dim_upsample_factor = 2
       
        if self.use_latent_concepts:
            precomputed_text_features = np.load(args.latent_concept_path)
        else:
            precomputed_text_features = np.load('./precomputed_features/solos_clip_rn50_text_category_features.npy')
        self.precomputed_text_features = torch.from_numpy(precomputed_text_features).float()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer6 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer7 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
        if 'textclass' in self.visual_arch or 'combined' in self.visual_arch:
            self.ce_loss = nn.CrossEntropyLoss()
            
        if 'mlp' in self.audio_arch:
            self.audio_mlp = nn.Sequential(nn.Linear(self.fc_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.fc_dim))
        
    def encode_audio(self, audio):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        return audio_conv7feature.squeeze()
        
    def compute_audio_text_sim(self, audio_feat, combined_cat_idx):
    
        norm_text = self.precomputed_text_features / self.precomputed_text_features.norm(dim=-1, keepdim=True)
        norm_text = norm_text.to(audio_feat.device)
        norm_aud = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
        scores = torch.matmul(norm_aud, norm_text.t())
        
        scores = scores * 100
        
        text_class_loss = self.ce_loss(scores, combined_cat_idx)
        return text_class_loss
        
    def compute_classification_loss(self, combined_pred_audio_feats, combined_cat_idx):
        text_class_loss = self.compute_audio_text_sim(combined_pred_audio_feats, combined_cat_idx)
        return text_class_loss
        
        
    def compute_text_classification(self, audio_mag_mix, pred_masks, first_cat_idx, second_cat_idx):
        combined_cat_idx = torch.cat((first_cat_idx, second_cat_idx), dim=0)
        combined_pred_mask = torch.cat((pred_masks[0], pred_masks[1]), dim=0)
        combined_mag_mix = audio_mag_mix.tile((len(pred_masks), 1, 1, 1))
        
        combined_pred_mag = combined_mag_mix * combined_pred_mask
        combined_aud_feature = self.encode_audio(combined_pred_mag)
        
        if 'mlp' in self.audio_arch:
            combined_aud_feature = self.audio_mlp(combined_aud_feature)
        
        text_class_loss = self.compute_audio_text_sim(combined_aud_feature, combined_cat_idx)
    
        return text_class_loss
        
    def encode_audio_spec(self, audio_mag_mix, pred_masks):
        pred_spec = audio_mag_mix * pred_masks
        pred_audio_feature = self.encode_audio(pred_spec)
    
        return pred_audio_feature
        
    def compute_audio_attn(self, audio_conv7feature, visual_feats, use_softmax):
        B, T, S, S, D = visual_feats.shape
        
        visual_feats = visual_feats.view(B, -1, D)
        
        norm_vis = visual_feats / visual_feats.norm(dim=-1, keepdim=True)
        norm_aud = audio_conv7feature / audio_conv7feature.norm(dim=-1, keepdim=True)
        
        scores = torch.bmm(norm_vis, norm_aud.unsqueeze(-1))
        
        scores = scores * 100
        
        if 'framewise' in self.visual_arch:
            scores = scores.view(B, T, -1)
        else:
            scores = scores.view(B, -1)
            
        if use_softmax:
            scores = F.softmax(scores, dim=-1)
        else:
            scores = F.log_softmax(scores, dim=-1)
            
        return scores
        
    def compute_audio_to_video_attn(self, pred_audio_feats, visual_feats, use_softmax=False):
        scores = self.compute_audio_attn(pred_audio_feats, visual_feats, use_softmax)
        return scores
        
        
    def encode_spec(self, x, visual_feats, use_softmax=False):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = F.adaptive_avg_pool2d(audio_conv7feature, 1)
        audio_conv7feature = audio_conv7feature.squeeze()
        
        if 'mlp' in self.audio_arch:
            audio_conv7feature = self.audio_mlp(audio_conv7feature)
        
        scores = self.compute_audio_attn(audio_conv7feature, visual_feats, use_softmax)
        return scores
        
        
    def upsample_conv(self, all_down_conv_features, feat, num_regions=None):
    
        if 'regionmask' in self.visual_arch:
            audio_upconv1feature = self.audionet_upconvlayer1(feat)
            audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[5].size(-3), all_down_conv_features[5].size(-2), all_down_conv_features[5].size(-1))), dim=1))
            audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[4].size(-3), all_down_conv_features[4].size(-2), all_down_conv_features[4].size(-1))), dim=1))
            audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[3].size(-3), all_down_conv_features[3].size(-2), all_down_conv_features[3].size(-1))), dim=1))
            audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[2].size(-3), all_down_conv_features[2].size(-2), all_down_conv_features[2].size(-1))), dim=1))
            audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[1].size(-3), all_down_conv_features[1].size(-2), all_down_conv_features[1].size(-1))), dim=1))
            mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0].unsqueeze(1).repeat(1, num_regions, 1, 1, 1).view(-1, all_down_conv_features[0].size(-3), all_down_conv_features[0].size(-2), all_down_conv_features[0].size(-1))), dim=1))
            mask_prediction = mask_prediction.view(-1, num_regions, mask_prediction.size(-3), mask_prediction.size(-2), mask_prediction.size(-1))
            
            mask_prediction = mask_prediction.sum(1)
            
            #mask_prediction = torch.max(mask_prediction, dim=1)[0]
            
            #print('mask_prediction: ', mask_prediction)
            #print('mask_prediction: ', mask_prediction.shape)
            #sys.exit()
            
            return mask_prediction
    
        audio_upconv1feature = self.audionet_upconvlayer1(feat)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5]), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4]), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3]), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2]), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1]), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0]), dim=1))
        return mask_prediction
        
    def forward_text(self, x, text):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = text[0]
        second_combined = text[1]
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction

    def forward(self, x, frame_feats):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        
        first_combined = frame_feats[0]
        second_combined = frame_feats[1]
        
        if 'regionmask' in self.visual_arch:
            first_combined = first_combined.view(first_combined.size(0), -1, first_combined.size(-1))
            second_combined = second_combined.view(second_combined.size(0), -1, second_combined.size(-1))
            
            audio_conv7feature = audio_conv7feature.unsqueeze(1).repeat(1, first_combined.size(1), 1, 1, 1)
            first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
            second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
            num_regions = first_combined.size(1)
            first_combined = torch.cat((first_combined, audio_conv7feature), dim=2)
            second_combined = torch.cat((second_combined, audio_conv7feature), dim=2)
            
            first_combined = first_combined.view(-1, first_combined.size(-3), first_combined.size(-2), first_combined.size(-1))
            second_combined = second_combined.view(-1, second_combined.size(-3), second_combined.size(-2), second_combined.size(-1))
            
            first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined, num_regions)
            second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined, num_regions)
           
            return first_mask_prediction, second_mask_prediction
            
        first_combined = first_combined.permute(0, 4, 1, 2, 3)
        second_combined = second_combined.permute(0, 4, 1, 2, 3)
        
        if 'maxpool' in self.visual_arch:
            first_combined = F.adaptive_max_pool3d(first_combined, 1)
            second_combined = F.adaptive_max_pool3d(second_combined, 1)
        else:
            first_combined = F.adaptive_avg_pool3d(first_combined, 1)
            second_combined = F.adaptive_avg_pool3d(second_combined, 1)
            
        first_combined = first_combined.squeeze()
        second_combined = second_combined.squeeze()
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        
        return first_mask_prediction, second_mask_prediction
        
class AudioVisual7layerUNet(nn.Module):
    def __init__(self, args, ngf=64, input_nc=1, output_nc=1):
        super(AudioVisual7layerUNet, self).__init__()
        
        self.args = args
        self.fc_dim = args.num_channels
        
        if self.fc_dim == 512:
            self.dim_upsample_factor = 1
        elif self.fc_dim == 1024:
            self.dim_upsample_factor = 2

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer6 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_convlayer7 = unet_conv(ngf * 8 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8 * self.dim_upsample_factor)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16 * self.dim_upsample_factor, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
    def upsample_conv(self, all_down_conv_features, feat):
        audio_upconv1feature = self.audionet_upconvlayer1(feat)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, all_down_conv_features[5]), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, all_down_conv_features[4]), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, all_down_conv_features[3]), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, all_down_conv_features[2]), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, all_down_conv_features[1]), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, all_down_conv_features[0]), dim=1))
        return mask_prediction
        
    def forward_text(self, x, text):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = text[0]
        second_combined = text[1]
        
        first_combined = first_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = second_combined.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction

    def forward(self, x, frame_feats):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        
        all_down_conv_features = [audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature, audio_conv5feature, audio_conv6feature]
        
        first_combined = frame_feats[0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        second_combined = frame_feats[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_conv7feature.size(-2), audio_conv7feature.size(-1))
        first_combined = torch.cat((first_combined, audio_conv7feature), dim=1)
        second_combined = torch.cat((second_combined, audio_conv7feature), dim=1)

        # compute masks for individual videos
        first_mask_prediction = self.upsample_conv(all_down_conv_features, first_combined)
        second_mask_prediction = self.upsample_conv(all_down_conv_features, second_combined)
        
        return first_mask_prediction, second_mask_prediction

class DebugSqueezeUnet(nn.Module):
    def __init__(self, args, audio_arch, fc_dim=64, num_downs=5, ngf=32, use_dropout=False, dropout=0.1):
        super(DebugSqueezeUnet, self).__init__()
        
        self.audio_arch = audio_arch
        self.bn0 = nn.BatchNorm2d(1)
        
        # downsample layers
        self.downconv_one = nn.Conv2d(
                1, ngf, kernel_size=4,
                stride=2, padding=1, bias=False)
                
        self.downsampling_factor = fc_dim // (ngf * 8)
        if fc_dim == 1024:
            self.downsampling_factor = self.downsampling_factor // 2

        self.downconv_two = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf, ngf*2))
        self.downconv_three = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 2, ngf * 4))
        self.downconv_four = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 4, ngf * 8))
        self.downconv_five = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 8, ngf * 8 * self.downsampling_factor))
        self.downconv_six = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 8 * self.downsampling_factor, fc_dim))
        self.downconv_final = nn.Sequential(*self.build_intermediate_squeeze_layer(fc_dim, fc_dim, innermost=True))
        
        # upsample layers
        self.upconv_one = nn.Sequential(*self.build_intermediate_upsample_layer(fc_dim, fc_dim*2, innermost=True))
        self.upconv_two = nn.Sequential(*self.build_intermediate_upsample_layer(fc_dim, fc_dim))
        self.upconv_three = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 8, fc_dim))
        self.upconv_four = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 4, ngf * 8))
        self.upconv_five = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 2, ngf * 4))
        self.upconv_six = nn.Sequential(*self.build_intermediate_upsample_layer(ngf, ngf * 2))
        self.upconv_final = nn.Sequential(*self.build_intermediate_upsample_layer(1, ngf, outermost=True))
        
        if 'attn' in self.audio_arch:
            self.multihead_attn = nn.MultiheadAttention(512, args.num_attention_heads, batch_first=True)
        elif 'trans' in self.audio_arch:
            encoder_layer = nn.TransformerEncoderLayer(512, args.num_attention_heads, args.ffn_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)
        
    def build_intermediate_upsample_layer(self, input_dim, output_dim, inner_output_nc=None, innermost=False, outermost=False):
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(input_dim)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
            
        if innermost:
            inner_output_nc = output_dim
        elif inner_output_nc is None:
            inner_output_nc = 2 * output_dim
            
        if innermost:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1, bias=False)
        elif not outermost:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1, bias=False)
        else:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1)
            
        if outermost:
            return [uprelu, upsample, upconv]
                    
        return [uprelu, upsample, upconv, upnorm]
        
    def build_intermediate_squeeze_layer(self, input_dim, output_dim, innermost=False):
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(output_dim)
        downconv = nn.Conv2d(
                input_dim, output_dim, kernel_size=4,
                stride=2, padding=1, bias=False)
                
        if innermost:
            return [downrelu, downconv]        
        
        return [downrelu, downconv, downnorm]
        
        
    def forward(self, x, frame_feats):
        x = self.bn0(x)
        down_first = self.downconv_one(x)
        down_second = self.downconv_two(down_first)
        down_third = self.downconv_three(down_second)
        down_fourth = self.downconv_four(down_third)
        down_fifth = self.downconv_five(down_fourth)
        down_sixth = self.downconv_six(down_fifth)
        down_final = self.downconv_final(down_sixth)
        
        num_regions = down_final.size(-1)
        if 'attn' in self.audio_arch:
            
            down_final = down_final.view(down_final.size(0), down_final.size(1), -1)
            down_final = down_final.permute(0, 2, 1).contiguous()
            down_final, _ = self.multihead_attn(down_final, down_final, down_final)
            down_final = down_final.permute(0, 2, 1)
            down_final = down_final.view(down_final.size(0), down_final.size(1), num_regions, num_regions)
        elif 'trans' in self.audio_arch:
            down_final = down_final.view(down_final.size(0), down_final.size(1), -1)
            down_final = down_final.permute(2, 0, 1).contiguous()
            down_final = self.transformer_encoder(down_final)
            down_final = down_final.permute(1, 2, 0).contiguous()
            down_final = down_final.view(down_final.size(0), down_final.size(1), num_regions, num_regions)
            
        first_combined = frame_feats[0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, down_final.size(-2), down_final.size(-1))
        second_combined = frame_feats[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, down_final.size(-2), down_final.size(-1))
        first_combined = torch.cat((down_final, first_combined), dim=1)
        second_combined = torch.cat((down_final, second_combined), dim=1)
                    
        up_first = self.upconv_one(first_combined)
        up_second = self.upconv_two(torch.cat((up_first, down_sixth), 1))
        up_third = self.upconv_three(torch.cat((up_second, down_fifth), 1))
        up_fourth = self.upconv_four(torch.cat((up_third, down_fourth), 1))
        up_fifth = self.upconv_five(torch.cat((up_fourth, down_third), 1))
        up_sixth = self.upconv_six(torch.cat((up_fifth, down_second), 1))
        up_final = self.upconv_final(torch.cat((up_sixth, down_first), 1))
               
        return up_final

class SqueezeUnet(nn.Module):
    def __init__(self, audio_arch, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(SqueezeUnet, self).__init__()
        
        self.audio_arch = audio_arch
        self.bn0 = nn.BatchNorm2d(1)
        
        # downsample layers
        self.downconv_one = nn.Conv2d(
                1, ngf, kernel_size=4,
                stride=2, padding=1, bias=False)

        self.downconv_two = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf, ngf*2))
        self.downconv_three = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 2, ngf * 4))
        self.downconv_four = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 4, ngf * 8))
        self.downconv_five = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 8, ngf * 8))
        self.downconv_six = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 8, ngf * 8))
        self.downconv_final = nn.Sequential(*self.build_intermediate_squeeze_layer(ngf * 8, ngf * 8, innermost=True))
        
        # upsample layers
        self.upconv_one = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 8, ngf * 8, innermost=True))
        self.upconv_two = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 8, ngf * 8))
        if 'upsample3' in self.audio_arch:
            self.upconv_three = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 8, ngf * 8))
            
        
        #self.upconv_four = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 4, ngf * 8))
        #self.upconv_five = nn.Sequential(*self.build_intermediate_upsample_layer(ngf * 2, ngf * 4))
        #self.upconv_six = nn.Sequential(*self.build_intermediate_upsample_layer(ngf, ngf * 2))
        #self.upconv_final = nn.Sequential(*self.build_intermediate_upsample_layer(fc_dim, ngf, outermost=True))
        
    def build_intermediate_upsample_layer(self, input_dim, output_dim, inner_output_nc=None, innermost=False, outermost=False):
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(input_dim)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
            
        if innermost:
            inner_output_nc = output_dim
        elif inner_output_nc is None:
            inner_output_nc = 2 * output_dim
            
        if innermost:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1, bias=False)
        elif not outermost:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1, bias=False)
        else:
            upconv = nn.Conv2d(
                inner_output_nc, input_dim, kernel_size=3,
                padding=1)
            
        if outermost:
            return [uprelu, upsample, upconv]
                    
        return [uprelu, upsample, upconv, upnorm]
        
    def build_intermediate_squeeze_layer(self, input_dim, output_dim, innermost=False):
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(output_dim)
        downconv = nn.Conv2d(
                input_dim, output_dim, kernel_size=4,
                stride=2, padding=1, bias=False)
                
        if innermost:
            return [downrelu, downconv]        
        
        return [downrelu, downconv, downnorm]
        
        
    def forward(self, x):
        x = self.bn0(x)
        down_first = self.downconv_one(x)
        down_second = self.downconv_two(down_first)
        down_third = self.downconv_three(down_second)
        down_fourth = self.downconv_four(down_third)
        down_fifth = self.downconv_five(down_fourth)
        down_sixth = self.downconv_six(down_fifth)
        down_final = self.downconv_final(down_sixth)
        
        up_first = self.upconv_one(down_final)
        up_second = self.upconv_two(torch.cat([down_sixth, up_first], 1))
        if 'upsample3' in self.audio_arch:
            up_third = self.upconv_three(torch.cat([down_fifth, up_second], 1))
            up_final = torch.cat([down_fourth, up_third], 1)
        else:
            up_final = torch.cat([down_fifth, up_second], 1)
            
        #up_first = self.upconv_one(down_final)
        #up_second = self.upconv_two(torch.cat([down_sixth, up_first], 1))
        #up_third = self.upconv_three(torch.cat([down_fifth, up_second], 1))
        #up_fourth = self.upconv_four(torch.cat([down_fourth, up_third], 1))
        #up_fifth = self.upconv_five(torch.cat([down_third, up_fourth], 1))
        #up_sixth = self.upconv_six(torch.cat([down_second, up_fifth], 1))
        #up_final = self.upconv_final(torch.cat([down_first, up_sixth], 1))
        
        '''print('input: ', x.shape)
        print('downsample first: ', down_first.shape)
        print('downsample second: ', down_second.shape)
        print('downsample third: ', down_third.shape)
        print('downsample fourth: ', down_fourth.shape)
        print('downsample fifth: ', down_fifth.shape)
        print('downsample sixth: ', down_sixth.shape)
        print('downsample final: ', down_final.shape)
        print('')
        print('upsample first: ', up_first.shape)
        print('upsample second: ', up_second.shape)
        print('upsample third: ', up_third.shape)
        print('upsample fourth: ', up_fourth.shape)
        print('upsample fifth: ', up_fifth.shape)
        print('upsample sixth: ', up_sixth.shape)
        print('upsample final: ', up_final.shape)
        sys.exit()'''
        
        return up_final
        
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class SqueezeUnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(SqueezeUnetBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
        #print(self.model)

    def forward(self, x):
        if self.outermost or self.noskip:
            tmp_x = self.model(x)
            print('outermost_x: ', tmp_x.shape)
            
            return tmp_x
            
            return self.model(x)
        else:
            tmp_x = self.model(x)
            cat_x = torch.cat([x, tmp_x], 1)
            print('x: ', x.shape)
            print('tmp_x: ', tmp_x.shape)
            print('cat_x: ', cat_x.shape)
            
            if self.innermost:
                print('innermost')
                #sys.exit()
            else:
                print('intermediate')
            print('')
            
            return cat_x
        
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(Unet, self).__init__()

        # construct unet structure
        unet_block = UnetBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block

    def forward(self, x):  
        x = self.bn0(x)
        x = self.unet_block(x)
        return x


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:            
            return self.model(x)
        else:        
            return torch.cat([x, self.model(x)], 1)
