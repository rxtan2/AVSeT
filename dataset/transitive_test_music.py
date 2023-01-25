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
        #self.bbox_anns = pickle.load(open('./gradcam_box_annotations/resnet50_base_object_boxes.pkl', 'rb'))
        #self.bbox_anns = pickle.load(open('./gradcam_box_annotations/resnet50_prompt_object_boxes.pkl', 'rb'))
        #self.bbox_anns = pickle.load(open('./gradcam_box_annotations/resnet50_prompt_person_object_boxes.pkl', 'rb'))
        self.bbox_anns = pickle.load(open('./gradcam_box_annotations/albef_all_boxes.pkl', 'rb'))
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
        
        # absolute frame/audio paths
        all_path_frames = []
        center_frameN = int(len(all_frames)) // 2
        for i in range(self.num_frames):
            idx_offset = (i - self.num_frames // 2) * self.stride_frames
            all_path_frames.append(os.path.join(vid_frame_dir, '{:06d}.jpg'.format(center_frameN + idx_offset)))
        all_frames = self._load_frames(all_path_frames)
        
        #center_frame = self._load_frames([center_frame_path])
        center_frame_idx = int(center_frame_path.split('/')[-1].split('.jpg')[0])
        center_time = (center_frame_idx - 0.5) / self.fps
        audio = self._load_audio(audio_path, center_time)
        
        ampN, phase_mix = self._stft(audio)
        audio_mags = ampN.unsqueeze(0)
        
        bbox = self.bbox_anns[vid_name][0]
        
        center_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        center_x = int(center_x / 16)
        center_y = int(center_y / 16)   
        bbox_center = torch.Tensor([center_x, center_y])

        ret_dict = {'frames': all_frames, 'mags': audio_mags, 'bbox_center': bbox_center, 'vid_name': [vid_name], 'phase_mix': phase_mix}

        return ret_dict
        
    def __len__(self):
        return len(self.clips)
