import os
import sys
import random
import pickle
import torch
from .base import BaseDataset


class UnseenMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(UnseenMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.mode = opt.mode
        
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
        if self.mode == 'curated_eval':
            self.curated_pairs = pickle.load(open('./data/curated_clean_zero_shot_eval_pairs.pkl', 'rb'))
            self.curated_vid2idx = pickle.load(open('./data/curated_clean_zero_shot_unseen_vid2idx.pkl', 'rb'))
        
        vid_dir = "/research/rxtan/object-detection/models/Sound-of-Pixels/data/full_unseen_frames/"
        self.vid2cat = {}
        for cat in os.listdir(vid_dir):
            cat_vids = os.listdir(os.path.join(vid_dir, cat))
            for vid in cat_vids:
                self.vid2cat[vid] = cat
                
        tmp = []
        self.discarded_classes = ['trombone', 'frenchhorn']
        for i in self.list_sample:
            vid = i[1]
            vid = vid.split('/')[-1]
            cat = self.vid2cat[vid]
            if cat not in self.discarded_classes:
                tmp.append(i)
        self.list_sample = tmp
        
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
                center_timeN = (center_frames[n] - 0.5) / self.fps 
                audios[n] = self._load_audio(path_audios[n], center_timeN)
                
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'bbox_centers': bbox_centers, 'clip_names': clip_names, 'present': present, 'path_frames': path_frames}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
