import os
import sys
import json
import random
import pickle
import torch
from .base import BaseDataset
import transformers
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class ClipTextSOLOSMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(ClipTextSOLOSMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix      
        self.arch_frame = opt.arch_frame  
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
        self._tokenizer = _Tokenizer()
        
        self.vid2cat = pickle.load(open('./solos_vid2cat.pkl', 'rb'))
        
    def get_center(self, bbox):
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / (16 * (384 / 224)))
        center_y = int(center_y / (16 * (384 / 224)))
        return center_x, center_y
        
    def get_tokens(self, text, context_length: int = 77, truncate: bool = True):
        '''encodings = self.tokenizer.encode_plus(text, add_special_tokens=True, truncation=True,
                                               max_length=self.max_len, padding="max_length",
                                               return_attention_mask=True, return_tensors="pt")
        inputs = encodings['input_ids'].squeeze()
        pad = torch.zeros((self.clip_max_len - len(inputs))).long()
        inputs = torch.cat((inputs, pad))              
        return inputs, encodings['attention_mask'].squeeze()'''
        
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
                
                if 'prompt' in self.arch_frame:
                     cat = 'a photo of a ' + cat
                cat = cat.lower()
                
                input_ids = self.get_tokens(cat)
                word_tokens[n] = input_ids
                
                center_timeN = (center_frames[n] - 0.5) / self.fps 
                audios[n] = self._load_audio(path_audios[n], center_timeN)
                
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'mags': mags, 'bbox_centers': bbox_centers, 'clip_names': clip_names, 'present': present, 'word_tokens': word_tokens, 'path_frames': path_frames}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
