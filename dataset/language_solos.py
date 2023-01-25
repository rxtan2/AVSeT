import os
import sys
import json
import random
import pickle
import torch
from .base import BaseDataset
import transformers
from transformers import RobertaTokenizer
transformers.utils.logging.set_verbosity(50)

class LanguageSOLOSMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(LanguageSOLOSMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix        
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.vid2cat = pickle.load(open('./solos_vid2cat.pkl', 'rb'))
        self.all_cats = set()
        for vid in self.vid2cat:
            self.all_cats.add(self.vid2cat[vid])
                
        max_len = 0
        for cat in self.all_cats:
          cat = self.tokenizer(cat.lower())['input_ids']
          if len(cat) > max_len:
            max_len = len(cat)
        self.max_len = max_len
        
    def get_center(self, bbox):
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / (16 * (384 / 224)))
        center_y = int(center_y / (16 * (384 / 224)))
        return center_x, center_y
        
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

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        bbox_centers = [None for n in range(N)]
        clip_names = [None for n in range(N)]
        present = [None for n in range(N)]
        word_tokens = [None for n in range(N)]
        word_masks = [None for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            infos[n] = self.get_negative(infos[0])

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
                
                clip_names[n] = tmp
                bbox_centers[n] = torch.Tensor([[first_center_x, first_center_y], [second_center_x, second_center_y]])
                present[n] = torch.tensor(second_present)
                frames[n] = self._load_frames(path_frames[n])
                
                cat = infoN[1]
                cat = cat.split('/')[-2]
                cat = cat.lower()
                
                cat = self.tokenizer.encode_plus(cat, padding='max_length', truncation=True, max_length=self.max_len, return_attention_mask=True, return_tensors='pt')
                word_tokens[n] = cat['input_ids'].squeeze()
                word_masks[n] = cat['attention_mask'].squeeze()
                
                center_timeN = (center_frames[n] - 0.5) / self.fps 
                audios[n] = self._load_audio(path_audios[n], center_timeN)
                
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'mags': mags, 'bbox_centers': bbox_centers, 'clip_names': clip_names, 'present': present, 'word_tokens': word_tokens, 'word_masks': word_masks, 'path_frames': path_frames}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
