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
    
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class LatentCodeDataset():
    def __init__(self, args):
        self.img_size = 224
        self.frame_process = _transform(self.img_size)
        self.video_dir = args.video_dir

        self._tokenizer = _Tokenizer()
        self.placeholder = 'word'
        self.num_learnable_embeddings = args.num_learnable_embeddings
        self.start_token_id = 49406
        self.end_token_id = 49407
        self.placeholder_id = 2653
        self.context_length = 77
        self.vid2cat = pickle.load(open(args.vid2cat_path), "rb"))
        self.cat2idx = {}
        
        valid_videos = pickle.load(open(args.valid_videos_path, 'rb'))
        
        self.video_list = []
        invalid = set()
        valid = set()
        count = 0
        for cat in os.listdir(self.video_dir):
            cat_dir = os.path.join(self.video_dir, cat)
            cat_videos = os.listdir(cat_dir)
            for curr_vid in cat_videos:                    
                curr_vid_name = curr_vid.split('.mp4')[0]
                
                if curr_vid_name not in self.vid2cat:
                    continue
                    
                if curr_vid not in valid_videos:
                    continue
                         
                curr_vid_path = os.path.join(cat_dir, curr_vid)
                
                vid = curr_vid_path.split('/')[-1]
                self.video_list.append(curr_vid_path)
                
                valid.add(vid)
        
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
            curr = Image.open(path)
            curr = self.frame_process(curr)
            frames.append(curr)  
        frames = torch.stack(frames)
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
        
    def get_negative(self, sample):
        first_vid = sample[1]
        first_vid = first_vid.split('/')[-1]
        first_cat = self.vid2cat[first_vid]
        
        while True:
            selected = random.randint(0, len(self.list_sample)-1)
            selected = self.list_sample[selected]
            selected_vid = selected[1]
            selected_vid = selected_vid.split('/')[-1]
            selected_cat = self.vid2cat[selected_vid]
            
            if selected_cat == first_cat:
                continue
            else:
                break
                
        return selected
        
    def get_tokens(self, text, context_length: int = 77, truncate: bool = True):      
        if isinstance(text, str):
            texts = [text]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        prompt_token = self._tokenizer.encoder["a photo of a "]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + [prompt_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
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
        
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.mp4')[0]
        
        cat = self.vid2cat[video_name].lower()
        cat_idx = torch.tensor(self.cat2idx[cat])
        
        video_frames = os.listdir(video_path)
        video_frames = sorted(video_frames)
        center_frameN = int(len(video_frames)) // 2
        
        center_frame_path = video_frames[center_frameN]
        center_frame_path = os.path.join(video_path, center_frame_path)
        
        frame = Image.open(center_frame_path)
        frame = self.frame_process(frame)
        
        # creates templated input_ids
        query = ''
        input_ids = self.get_tokens(query)
        
        first = input_ids[:1]
        second = input_ids[1:]
        placeholder_input_ids = torch.zeros(self.num_learnable_embeddings).long()
        placeholder_input_ids += self.placeholder_id
        final_placeholder_input_ids = torch.cat((first, placeholder_input_ids, second))
        final_placeholder_input_ids = final_placeholder_input_ids[:self.context_length]
        
        ret_dict = {'text': final_placeholder_input_ids, 'frame': frame, 'video_name': video_name, 'cat_idx': cat_idx}
        
        return ret_dict
