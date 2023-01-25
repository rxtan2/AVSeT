import os
import sys
import random
import pickle
import json
import torch
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix

        self.bbox_anns = pickle.load(open("/research/rxtan/object-detection/models/Sound-of-Pixels/gradcam_box_annotations/albef_dual_boxes.pkl", 'rb'))
        self.clips = list(self.bbox_anns.keys())
        
        json_files = ["/research/rxtan/object-detection/datasets/MUSIC_dataset/MUSIC21_solo_videos.json", "/research/rxtan/object-detection/datasets/MUSIC_dataset/MUSIC_duet_videos.json", "/research/rxtan/object-detection/datasets/MUSIC_dataset/MUSIC_solo_videos.json"]

        self.vid2cat = {}
        all_videos = set()

        for curr_file in json_files:
            data = json.load(open(curr_file))
            videos = data['videos']
            for cat in videos:
                cat_videos = videos[cat]
                for vid in cat_videos:
                    self.vid2cat[vid] = cat
                    
        self.frame_dir = './data/frames/'
        self.audio_dir = './data/audio/'

    def __getitem__(self, index):
    
        vid_name = self.clips[index]
        vid_cat = self.vid2cat[vid_name]
        if ' ' in vid_cat:
            vid_cat = vid_cat.replace(' ', '_____')
            
        vid_frame_dir = os.path.join(self.frame_dir, vid_cat, '%s.mp4' % vid_name)
        all_frames = os.listdir(vid_frame_dir)
        center_frame_path = os.path.join(vid_frame_dir, all_frames[len(all_frames) // 2])
        audio_path = os.path.join(self.audio_dir, vid_cat, '%s.mp3' % vid_name)
        
        center_frame = self._load_frames([center_frame_path])
        center_frame_idx = int(center_frame_path.split('/')[-1].split('.jpg')[0])
        center_time = (center_frame_idx - 0.5) / self.fps
        audio = self._load_audio(audio_path, center_time)
        
        ampN, phase_mix = self._stft(audio)
        audio_mags = ampN.unsqueeze(0)
        
        boxes = self.bbox_anns[vid_name]
        center_boxes = []
        
        for bbox in boxes:
            center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
            center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
            center_x = int(center_x / (16 * (384 / 224)))
            center_y = int(center_y / (16 * (384 / 224)))
            center_boxes.append([center_x, center_y])
        center_boxes = torch.tensor(center_boxes)

        ret_dict = {'frames': center_frame, 'mags': audio_mags, 'bbox_center': center_boxes, 'vid_name': [vid_name], 'phase_mix': phase_mix}

        return ret_dict
        
    def __len__(self):
        return len(self.clips)
