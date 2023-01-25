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

class ClipJointMUSICPositionMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(ClipJointMUSICPositionMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.img_size = 224
        self.split = kwargs['split']
        self.mode = opt.mode
        self.text_embeddings_path = opt.text_embeddings_path
        self.text_cat_order_path = opt.text_cat_order_path
        
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
        
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
                
        self.precomputed_text_features = np.load(self.text_embeddings_path)
        tmp = pickle.load(open(self.text_cat_order_path, 'rb'))
        
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
                
        self.cat2vids = {}
        self.invalid_indices = set()
        for idx, i in enumerate(self.list_sample):
            vid = i[0].split('/')[-1].split('.wav')[0]
            vid_cat = self.vid2cat[vid]
            if vid_cat not in self.cat2vids:
                self.cat2vids[vid_cat] = set()
            self.cat2vids[vid_cat].add((vid, idx, ))  
        
        
        for i in self.cat2vids:
            tmp = self.cat2vids[i]
            if len(tmp) < 2:
                for j in tmp:
                    self.invalid_indices.add(j[-1])
                    
        tmp = []
        for idx, i in enumerate(self.list_sample):
            if idx not in self.invalid_indices:
                tmp.append(i)
        self.list_sample = tmp
        
        self.cat2vids = {}
        for idx, i in enumerate(self.list_sample):
            vid = i[0].split('/')[-1].split('.wav')[0]
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
        
    def _load_clip_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(Image.open(path).convert('RGB'))
        #frames = self.preprocess(frames)
        return frames
        
    def get_same_instrument_video(self, info):
        video = info[0]
        video = video.split('/')[-1]
        video = video.split('.wav')[0]
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

    def get_concat_v(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
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
        combined_images = []

        # the first video
        infos[0] = self.list_sample[index]
        
        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            infos[n] = self.get_same_instrument_video(infos[0])

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
                cat = self.vid2cat[tmp]
                cat = cat.replace(' ', ' and ')
                cat = cat.replace('_', ' ')
                
                if 'name' in self.text_embeddings_path:
                    cat_idx = self.cat2text_feat_idx[cat]
                    text_feat = self.precomputed_text_features[cat_idx]
                    text[n] = text_feat[n]
                else:
                    text[n] = self.precomputed_text_features[n]
                
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
                
            # concatenate frames
            concatenated_frames = []
            for n in range(len(frames[0])):
                curr = self.get_concat_h(frames[0][n].resize((224, 224)), frames[1][n].resize((224, 224)))
                concatenated_frames.append(curr)
                combined_images.append(transforms.ToTensor()(curr))
            
            concatenated_frames = self.preprocess(concatenated_frames)
            frames[0] = concatenated_frames
            frames[1] = concatenated_frames.clone()
                            
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'text': text, 'frames': frames, 'mags': mags, 'clip_names': clip_names, 'bbox_centers': bbox_centers, 'path_frames': path_frames, 'combined_images': combined_images}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
