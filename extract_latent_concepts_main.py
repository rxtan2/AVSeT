# System libs
import os
import sys
import random
import time
import argparse

# Numerical libs
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
#from scipy.misc import imsave

# comment out for training on scc
import pydub
from models.vision_net import *
from models.clip_latent_models import *

from PIL import Image
import cv2
import torchvision.transforms.functional as VF
import pickle

# Our libs
from arguments import ArgParser
from dataset import LatentCodeDataset

# tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Model to extract latent code representations')
parser.add_argument('--batch_size', type=int, default=58, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--workers', type=int, default=5, metavar='N',
                    help='number of worker threads')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='Number of training epochs (default: 20)')
parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N',
                    help='Number of warmup steps')
parser.add_argument('--num_learnable_embeddings', type=int, default=3, metavar='NE',
                    help='Number of learnable embeddings for latent concepts')
parser.add_argument('--lr', type=float, default=10., metavar='LR',
                    help='learning rate for optimizer')
parser.add_argument('--video_dir', type=str, default="/research/rxtan/object-detection/models/Sound-of-Pixels/data/audioset_dataset_frames/", metavar='VD', help='video directory')

def main(args):
    dataset = LatentCodeDataset(args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)
        
    '''loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False)'''
        
    model = ClipLatentModel()
    model = model.cuda()
    model.eval()
    train(args, model, loader, args.epochs, args.lr)
    
def train(args, model, loader, num_epochs, lr):
    all_latent_concepts = []
    all_frame_features = []
    video_order = []
    
    for batch_idx, data in enumerate(loader):
        print(batch_idx)
        text = data['text'].cuda()
        frame = data['frame'].cuda()
        cat_idx = data['cat_idx'].cuda()
        
        video_order += data['video_name']
        
        # initialize latent concepts
        concept_model = LatentConceptEmbedding(len(text), args.num_learnable_embeddings).cuda()
        concept_model.train()
        
        #optimizer = torch.optim.Adam(concept_model.parameters(), lr)
        optimizer = torch.optim.SGD(concept_model.parameters(), lr)
        
        with torch.no_grad():
            visual_features = concept_model.clip_model.encode_image(frame)
            text_features = concept_model.clip_model.get_token_embeddings(text)
        
        initial = concept_model.latent_concept.clone().detach()
        tmp = concept_model.latent_concept.clone().detach()
        
        #tmp = F.normalize(tmp, dim=-1)
        #initial = F.normalize(initial, dim=-1)
        #initial_dist = torch.cdist(initial.unsqueeze(0), tmp.unsqueeze(0))
        #initial_dist = initial_dist * torch.eye(len(initial_dist)).cuda()
        #initial_dist = initial_dist.sum(1)
        
        for curr_epoch in range(num_epochs):
            optimizer.zero_grad()
            score, text_class_scores, gt_class_scores = concept_model(text, text_features, visual_features, cat_idx)
            score = score * -1.
            score.backward()
            optimizer.step()
            
            #print('score %s: ' % curr_epoch, score.item(), ' , ', 'text class acc: ', text_class_scores.item(), ' , ', 'gt_text_class score: ', gt_class_scores.item())
            
        #sys.exit()
            
        #final = concept_model.latent_concept.clone().detach()
        #final = F.normalize(final, dim=-1)
        #initial = F.normalize(initial, dim=-1)
        #dist = torch.cdist(initial.unsqueeze(0).unsqueeze(0), final.unsqueeze(0).unsqueeze(0))
        
        with torch.no_grad():
            latent_concept = concept_model.extract_latent_concept(text, text_features)
            latent_concept = latent_concept.squeeze(0).detach().cpu()
            
        all_latent_concepts.append(latent_concept)
        all_frame_features.append(visual_features.detach().cpu())
        
    all_latent_concepts = torch.cat(all_latent_concepts, dim=0)
    all_frame_features = torch.cat(all_frame_features, dim=0)
    all_latent_concepts = all_latent_concepts.numpy()
    all_frame_features = all_frame_features.numpy()
    
    np.save('./precomputed_features/latent_features/audioset_split_5_%s_latent_concept_sgd_epoch_%s_lr_%s_model.npy' % (args.num_learnable_embeddings, args.epochs, lr), all_latent_concepts)
    #np.save('./precomputed_features/latent_features/visual_solos_%s_latent_concept_sgd_epoch_%s_lr_%s_model.npy' % (args.num_learnable_embeddings, args.epochs, lr), all_frame_features)
    pickle.dump(video_order, open('./precomputed_features/latent_features/audioset_split_5_%s_latent_concept_sgd_epoch_%s_lr_%s_model.pkl' % (args.num_learnable_embeddings, args.epochs, lr), 'wb'))
    
    #np.save('./precomputed_features/latent_features/person_template_solos_%s_latent_concept_sgd_epoch_%s_lr_%s_model.npy' % (args.num_learnable_embeddings, args.epochs, lr), all_latent_concepts)
    #np.save('./precomputed_features/latent_features/visual_photo_person_template_solos_%s_latent_concept_sgd_epoch_%s_lr_%s_model.npy' % (args.num_learnable_embeddings, args.epochs, lr), all_frame_features)
    #pickle.dump(video_order, open('./precomputed_features/latent_features/person_template_solos_%s_latent_concept_sgd_epoch_%s_lr_%s_model.pkl' % (args.num_learnable_embeddings, args.epochs, lr), 'wb'))
    
    print('all_latent_concepts: ', all_latent_concepts.shape)
    print('video_order: ', len(video_order))
    
    return

if __name__ == '__main__':
    # arguments
    global args
    args = parser.parse_args()
    
    main(args)
