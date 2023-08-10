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
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for optimization (default: 16)')
parser.add_argument('--workers', type=int, default=5, metavar='N',
                    help='number of worker threads')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='Number of training epochs (default: 5000)')
parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N',
                    help='Number of warmup steps')
parser.add_argument('--num_learnable_embeddings', type=int, default=75, metavar='NE',
                    help='Number of learnable embeddings for latent concepts')
parser.add_argument('--lr', type=float, default=10., metavar='LR',
                    help='learning rate for optimizer')
parser.add_argument('--video_dir', type=str, default="./video_dataset_frames/", metavar='VD', help='video directory')
parser.add_argument('--output_latents_feature_path', type=str, default="./latent_embeds.npy", metavar='VD', help='path to save latent embeddings')
parser.add_argument('--output_latents_order_path', type=str, default="./latent_embeds_order.pkl", metavar='VD', help='path to save order of latent embeddings')

def main(args):
    dataset = LatentCodeDataset(args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)
  
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
        
        optimizer = torch.optim.SGD(concept_model.parameters(), lr)
        
        with torch.no_grad():
            visual_features = concept_model.clip_model.encode_image(frame)
            text_features = concept_model.clip_model.get_token_embeddings(text)
        
        initial = concept_model.latent_concept.clone().detach()
        tmp = concept_model.latent_concept.clone().detach()
        
        for curr_epoch in range(num_epochs):
            optimizer.zero_grad()
            score, text_class_scores, gt_class_scores = concept_model(text, text_features, visual_features, cat_idx)
            score = score * -1.
            score.backward()
            optimizer.step()

        with torch.no_grad():
            latent_concept = concept_model.extract_latent_concept(text, text_features)
            latent_concept = latent_concept.squeeze(0).detach().cpu()
            
        all_latent_concepts.append(latent_concept)
        all_frame_features.append(visual_features.detach().cpu())
        
    all_latent_concepts = torch.cat(all_latent_concepts, dim=0)
    all_frame_features = torch.cat(all_frame_features, dim=0)
    all_latent_concepts = all_latent_concepts.numpy()
    all_frame_features = all_frame_features.numpy()
    
    np.save(args.output_latents_feature_path, all_latent_concepts)
    pickle.dump(video_order, open(args.output_latents_order_path, 'wb'))  
    return

if __name__ == '__main__':
    # arguments
    global args
    args = parser.parse_args()
    
    main(args)
