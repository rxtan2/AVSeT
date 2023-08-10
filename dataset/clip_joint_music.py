import os
import sys
import json
import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
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
    
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def train_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(p=0.5),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ClipJointMUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(ClipJointMUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.img_size = 224
        self.split = kwargs['split']
        self.mode = opt.mode
        self.visual_arch = opt.arch_frame
        self.use_latent_concepts = opt.use_latent_concepts
        
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
                
        if 'randomaug' in self.visual_arch and self.split == 'train':
            self.frame_process = train_transform(self.img_size)
        else:
            self.frame_process = _transform(self.img_size)

        if self.use_latent_concepts:
            self.latent_concept_path = opt.latent_concept_path
            self.precomputed_latent_features = np.load(self.latent_concept_path)
            tmp = pickle.load(open(self.latent_concept_path.replace('npy', 'pkl'), 'rb'))
            self.vid2latent_idx = {}
            for idx, vid in enumerate(tmp):
                self.vid2latent_idx[vid] = idx
        
    def get_center(self, bbox):
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / (16 * (384 / 224)))
        center_y = int(center_y / (16 * (384 / 224)))
        return center_x, center_y
        
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
        first_vid_name = first_vid.split('.mp4')[0]
        first_cat = self.vid2cat[first_vid_name]
        
        while True:
            selected = random.randint(0, len(self.list_sample)-1)
            selected = self.list_sample[selected]
            selected_vid = selected[1]
            selected_vid = selected_vid.split('/')[-1]
            selected_cat = self.vid2cat[selected_vid.split('.mp4')[0]]
            
            if selected_cat == first_cat:
                continue
            else:
                break
                
        return selected

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
        cat_pair_idx = [None for n in range(N)]
        
        if self.use_latent_concepts:
            latent_pair_idx = [None for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        if self.mode != 'curated_eval':
            for n in range(1, N):                
                infos[n] = self.get_negative(infos[0])
        else:
            vid_name = infos[0][0]
            vid_name = vid_name.split('/')[-1]
            vid_name = vid_name.split('.wav')[0]
            
            second_vid_name = self.curated_pairs[vid_name]
            second_vid_idx = self.curated_vid2idx[second_vid_name]
            infos[1] = self.list_sample[second_vid_idx]

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                tmp = path_frames[n][0]
                tmp = tmp.split('/')[-2]
                tmp = tmp.split('.mp4')[0]
                
                cat = self.vid2cat[tmp]
                cat = cat.replace(' ', ' and ')
                cat = cat.replace('_', ' ')
                cat_idx = self.cat2text_feat_idx[cat]
                
                if self.use_latent_concepts:
                    curr_vid_name = path_frames[n][0].split('/')[-2]
                    vid_latent_idx = self.vid2latent_idx[curr_vid_name.split('.mp4')[0]]
                    text_feat = self.precomputed_latent_features[vid_latent_idx]
                    latent_pair_idx[n] = vid_latent_idx
                else:
                    text_feat = self.precomputed_text_features[cat_idx]
                
                text[n] = text_feat
                
                # gets albef-computed bounding boxes
                if tmp in self.bbox_anns:
                    bbox = self.bbox_anns[tmp]
                else:
                    bbox = [[0, 0, 0, 0], None]    
                 
                first_box = bbox[0]
                second_box = bbox[1]  
                first_center_x, first_center_y = self.get_center(first_box)
                if second_box is not None:
                    second_present = True
                    second_center_x, second_center_y = self.get_center(second_box)
                else:
                    second_present = False
                    second_center_x = 0
                    second_center_y = 0
                bbox_centers[n] = torch.Tensor([[first_center_x, first_center_y], [second_center_x, second_center_y]])
                
                clip_names[n] = tmp
                frames[n] = self._load_clip_frames(path_frames[n])
                
                center_timeN = (center_frames[n] - 0.5) / self.fps 
                audios[n] = self._load_audio(path_audios[n], center_timeN)
                
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)
                
        if self.use_latent_concepts:
            first_latent_idx = latent_pair_idx[0]
            second_latent_idx = latent_pair_idx[1]
            ret_dict = {'mag_mix': mag_mix, 'text': text, 'frames': frames, 'mags': mags, 'clip_names': clip_names, 'bbox_centers': bbox_centers, 'path_frames': path_frames, 'first_cat_idx': first_latent_idx, 'second_cat_idx': second_latent_idx}
            
            if self.split != 'train':
                ret_dict['audios'] = audios
                ret_dict['phase_mix'] = phase_mix
                ret_dict['infos'] = infos

            return ret_dict

        ret_dict = {'mag_mix': mag_mix, 'text': text, 'frames': frames, 'mags': mags, 'clip_names': clip_names, 'bbox_centers': bbox_centers, 'path_frames': path_frames}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
