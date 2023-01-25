import os
import sys
import json
import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .base import BaseDataset
from PIL import Image
from . import video_transforms as vtransforms
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class ClipPOSAttnSOLOSMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(ClipPOSAttnSOLOSMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.img_size = 224
        self.split = kwargs['split']
        self.mode = opt.mode
        
        transform_list = []
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
        transform_list.append(vtransforms.CenterCrop(self.imgSize))
        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.preprocess = transforms.Compose(transform_list)          
          
        self.vid2cat = pickle.load(open('./solos_vid2cat.pkl', 'rb'))
        self._tokenizer = _Tokenizer()
        
        self.cat2vids = {}
        for idx, i in enumerate(self.list_sample):
            vid = i[1].split('/')[-1]
            vid_cat = self.vid2cat[vid]
            if vid_cat not in self.cat2vids:
                self.cat2vids[vid_cat] = set()
            self.cat2vids[vid_cat].add((vid, idx, ))  
        
    def get_center(self, bbox):
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / (16 * (384 / 224)))
        center_y = int(center_y / (16 * (384 / 224)))
        return center_x, center_y
        
    def get_tokens(self, text, context_length: int = 77, truncate: bool = True):        
        if isinstance(text, str):
            texts = [text]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
        result = result.squeeze(0)
        
        pad = torch.zeros((context_length - len(result))).long()
        result = torch.cat((result, pad))           
        return result
        
    def _load_clip_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(Image.open(path).convert('RGB'))
        #frames = self.preprocess(frames)
        return frames
        
    def get_different_instrument_video(self, info):
        video = info[0]
        video = video.split('/')[-1]
        video = video.split('.wav')[0]
        video_cat = self.vid2cat[video]
        
        selected = None
        while True:
            indexN = random.randint(0, len(self.list_sample)-1)
            selected = self.list_sample[indexN]
            
            if selected[0] != info[0]:
                break
        return selected
        
    def get_same_instrument_video(self, info):
        video = info[1]
        video = video.split('/')[-1]
        video_cat = self.vid2cat[video]
        
        all_cat_videos = self.cat2vids[video_cat]
        
        selected = None
        while True:
            selected = random.sample(all_cat_videos, 1)[0]
            if selected[0] == video:
                continue
            else:
                break
                
        return self.list_sample[selected[1]]        
    
    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def __getitem__(self, index):
        N = self.num_mix
        text = [None for n in range(N)]
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        bbox_centers = [None for n in range(N)]
        clip_names = [None for n in range(N)]
        present = [None for n in range(N)]

        # the first video
        path_audioN, path_frameN, count_framesN = self.list_sample[index]
        center_frameN = int(count_framesN) // 2
        
        # the second video
        second_path_audioN, second_path_frameN, second_count_framesN = self.get_same_instrument_video(self.list_sample[index])
        second_center_frameN = int(second_count_framesN) // 2
        
        path_frames = []
        second_path_frames = []
        for i in range(self.num_frames):
            idx_offset = (i - self.num_frames // 2) * self.stride_frames
            path_frames.append(os.path.join(path_frameN,'{:06d}.jpg'.format(center_frameN + idx_offset)))
            second_path_frames.append(os.path.join(second_path_frameN,'{:06d}.jpg'.format(second_center_frameN + idx_offset)))
            
        first_frames = self._load_clip_frames(path_frames)
        second_frames = self._load_clip_frames(second_path_frames)
            
        # concatenate frames
        concatenated_frames = []
        for n in range(len(first_frames)):
            curr = self.get_concat_h(first_frames[n].resize((256, 256)), second_frames[n].resize((256, 256)))
            concatenated_frames.append(curr)
        concatenated_frames = self.preprocess(concatenated_frames)
            
        tmp = path_frameN.split('/')
        vid = tmp[-1]
        cat = tmp[-2]
        vid = vid.split('.mp4')[0]  
        cat = cat.lower()
        
        #first_query = 'left ' + cat
        #second_query = 'right ' + cat
        
        first_query = cat + ' on the left'
        second_query = cat + ' on the right'

        #query = 'a video of a person playing a' + cat
        
        first_query_tokens = self.get_tokens(first_query)
        second_query_tokens = self.get_tokens(second_query)
        
        ret_dict = {'first_text': first_query_tokens, 'second_text': second_query_tokens, 'frames': concatenated_frames, 'query': [first_query], 'path_frames': path_frames, 'second_path_frames': second_path_frames}
        
        return ret_dict
