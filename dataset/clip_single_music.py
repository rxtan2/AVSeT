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

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class ClipDuetMUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(ClipDuetMUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.img_size = 224
        self.split = kwargs['split']
        self.mode = opt.mode
        
        transform_list = []
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.img_size * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.img_size))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))
        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.preprocess = transforms.Compose(transform_list)          

        if self.mode == 'curated_eval':
            self.curated_pairs = pickle.load(open('./data/curated_clean_zero_shot_eval_pairs.pkl', 'rb'))
            self.curated_vid2idx = pickle.load(open('./data/curated_clean_zero_shot_unseen_vid2idx.pkl', 'rb'))
        
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
        
        if 'vitb32' not in opt.arch_frame:
            if 'prompt' in opt.arch_frame:
                self.precomputed_text_features = np.load('./precomputed_features/prompt_clip_rn50_text_category_features.npy')
                tmp = pickle.load(open('./precomputed_features/prompt_text_category_order_rn50.pkl', 'rb'))
            else:
                self.precomputed_text_features = np.load('./precomputed_features/clip_text_category_features.npy')
                tmp = pickle.load(open('./precomputed_features/text_category_order.pkl', 'rb'))
        else:
            if 'prompt' in opt.arch_frame:
                self.precomputed_text_features = np.load('./precomputed_features/prompt_clip_vitb32_text_category_features.npy')
                tmp = pickle.load(open('./precomputed_features/prompt_text_category_order_vitb32.pkl', 'rb'))
            else:
                self.precomputed_text_features = np.load('./precomputed_features/clip_vitb32_text_category_features.npy')
                tmp = pickle.load(open('./precomputed_features/text_category_order_vitb32.pkl', 'rb'))
        
        self.cat2text_feat_idx = {}
        for idx, cat in enumerate(tmp):
          self.cat2text_feat_idx[cat] = idx
          
        json_file_dir = '/research/rxtan/object-detection/datasets/MUSIC_dataset/'
        if not os.path.exists(json_file_dir):
            json_file_dir = '/net/ivcfs4/mnt/data/rxtan/object-detection/datasets/MUSIC_dataset/'
        
        json_file_names = ["MUSIC21_solo_videos.json", "MUSIC_duet_videos.json", "MUSIC_solo_videos.json"]
        json_files = [os.path.join(json_file_dir, curr_file) for curr_file in json_file_names]

        self.vid2cat = {}
        self.all_cats = set()
        for curr_file in json_files:
            data = json.load(open(curr_file))
            videos = data['videos']
            for cat in videos:
                cat_videos = videos[cat]
                for vid in cat_videos:
                    self.vid2cat[vid] = cat
                self.all_cats.add(cat)
              
        tmp = []  
        for i in self.list_sample:
            vid = i[0].split('/')[-1]
            if '.mp3' in vid:
                vid = vid.split('.mp3')[0]
            elif '.wav' in vid:
                vid = vid.split('.wav')[0]
            vid_cat = self.vid2cat[vid]
            if ' ' in vid_cat:
                tmp.append(i)
        self.list_sample = tmp
        
    def get_center(self, bbox):
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / (16 * (384 / 224)))
        center_y = int(center_y / (16 * (384 / 224)))
        return center_x, center_y
        
    def _load_clip_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(Image.open(path).convert('RGB'))
            
        frames = self.preprocess(frames)
        return frames

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
        infos[0] = self.list_sample[index]
        infos[1] = self.list_sample[index]

        # select frames
        idx_margin = max(int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        path_audioN, path_frameN, count_framesN = infos[0]
        center_frameN = int(count_framesN) // 2
        center_frames[0] = center_frameN
        center_frames[1] = center_frameN
        
        # absolute frame/audio paths
        for i in range(self.num_frames):
            idx_offset = (i - self.num_frames // 2) * self.stride_frames
            path_frames[0].append(os.path.join(path_frameN, '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_frames[1].append(os.path.join(path_frameN, '{:06d}.jpg'.format(center_frameN + idx_offset)))
            
        path_audios[0] = path_audioN
        path_audios[1] = path_audioN

        # gets text features
        tmp = path_frames[0][0]
        tmp = tmp.split('/')[-2]
        tmp = tmp.split('.mp4')[0]
                
        cat = self.vid2cat[tmp]
        cat = self.vid2cat[tmp]
        cat = cat.split(' ')
        first_cat = cat[0]
        first_cat = first_cat.replace('_', ' ')
        second_cat = cat[1]
        second_cat = second_cat.replace('_', ' ')
        
        first_cat_idx = self.cat2text_feat_idx[first_cat]
        second_cat_idx = self.cat2text_feat_idx[second_cat]
        
        text[0] = self.precomputed_text_features[first_cat_idx]
        text[1] = self.precomputed_text_features[second_cat_idx]
                
        # load frames and audios, STFT
        
        frames[0] = self._load_clip_frames(path_frames[0])
        frames[1] = self._load_clip_frames(path_frames[1])
        
        center_timeN = (center_frames[0] - 0.5) / self.fps 
        audios[0] = self._load_audio(path_audios[0], center_timeN)
        audios[1] = self._load_audio(path_audios[1], center_timeN)
        
        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
        
        clip_names[0] = tmp
        clip_names[1] = tmp

        ret_dict = {'mag_mix': mag_mix, 'text': text, 'frames': frames, 'mags': mags, 'clip_names': clip_names, 'path_frames': path_frames}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
