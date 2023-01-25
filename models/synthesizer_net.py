import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleInnerProd(nn.Module):
    def __init__(self, arch, fc_dim):
        super(UpsampleInnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.arch = arch
        
        if 'fc' in self.arch:
            if 'upsample3' in self.arch:
                self.upsample_fc = nn.Linear(16**2, 256**2)
            elif 'upsample2' in self.arch:
                self.upsample_fc = nn.Linear(8**2, 256**2)

    def forward(self, feat_img, feat_sound):
    
        #print('feat_img: ', feat_img.shape)
        
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            #.view(B, 1, *sound_size[2:])
        
        #print('output: ', z.shape)
        
        z = z.view(B, 1, *sound_size[2:])
        
        #print('view output: ', z.shape)      
        
        z = z + self.bias
        
        #print('bias output: ', z.shape)
        #sys.exit() 
        
        if 'fc' in self.arch:
            num_regions = z.size(-1)
            z = z.view(z.size(0), z.size(1), -1)
            z = self.upsample_fc(z)
            num_regions = int(z.size(-1) ** 0.5)
            z = z.view(z.size(0), z.size(1), num_regions, -1)
            
        return z

class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
    
        #print('feat_img: ', feat_img.shape)
        
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            #.view(B, 1, *sound_size[2:])
        
        #print('output: ', z.shape)
        
        z = z.view(B, 1, *sound_size[2:])
        
        #print('view output: ', z.shape)      
        
        z = z + self.bias
        
        #print('bias output: ', z.shape)
        #sys.exit()
        
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        
        print('feats_img: ', feats_img.shape)
        print('feat_sound: ', feat_sound.shape)
        
        feats_img = feats_img.view(B, C, HI*WI)
        
        #print('view feats_img: ', feats_img.shape)
        
        feats_img = feats_img.transpose(1, 2)
        
        #print('transpose feats_img: ', feats_img.shape)
        
        feat_sound = feat_sound.view(B, C, HS * WS)
        
        #print('view feats_sound: ', feat_sound.shape)
        
        z = torch.bmm(feats_img * self.scale, feat_sound) \
            .view(B, HI, WI, HS, WS)
            
        #print('z: ', z.shape)
            
        z = z + self.bias
        
        #print('bias z: ', z.shape)
        #sys.exit()
        
        return z


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # self.bias = nn.Parameter(-torch.ones(1))

    def forward(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z
