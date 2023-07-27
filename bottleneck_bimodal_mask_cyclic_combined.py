# System libs
import os
import sys
import random
import time

# Numerical libs
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
#from scipy.misc import imsave

# comment out for training on scc
from mir_eval.separation import bss_eval_sources
import pydub

from PIL import Image
import cv2

# Our libs
from arguments import ArgParser
from dataset import ClipJointMUSICMixDataset, ClipJointSOLOSMixDataset, ClipJointUnseenMixDataset, ClipJointAudioSetMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer

# tensorboard
from torch.utils.tensorboard import SummaryWriter

def log(path, output):
    with open(path, "a") as f:
        f.write(output + '\n')


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit, args):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame = nets
        self.crit = crit
        self.visual_arch = args.arch_frame
        self.kl_loss_weight = args.kl_loss_weight
        self.textclass_loss_weight = args.textclass_loss_weight
        self.text_mask_pred_loss_weight = args.text_mask_pred_loss_weight
        self.visual_mask_pred_loss_weight = args.visual_mask_pred_loss_weight

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        text = batch_data['text']
        bbox_centers = batch_data['bbox_centers']
        mag_mix = mag_mix + 1e-10
        first_cat_idx = batch_data['first_cat_idx'].cuda()
        second_cat_idx = batch_data['second_cat_idx'].cuda()

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()
        
        # 1. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        attn_scores = [None for n in range(N)]
        for n in range(N):
            feat_frames[n], attn_scores[n] = self.net_frame.forward_multiframe(frames[n], text[n])
        
        # 2. forward audio-visual unet
        pred_masks = [None for n in range(N)]
        text_pred_masks = [None for n in range(N)]
        pred_masks[0], pred_masks[1], text_pred_masks[0], text_pred_masks[1] = self.net_sound(log_mag_mix, feat_frames, text)
        
        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)
        text_err = self.crit(text_pred_masks, gt_masks, weight).reshape(1)
        
        # Encodes predicted audio spectrograms
        first_pred_audio_feats = self.net_sound.encode_audio_spec(log_mag_mix, pred_masks[0])
        second_pred_audio_feats = self.net_sound.encode_audio_spec(log_mag_mix, pred_masks[1])
        combined_pred_audio_feats = torch.cat((first_pred_audio_feats, second_pred_audio_feats), dim=0)
        
        # Compute classification loss
        if self.textclass_loss_weight > 0.0:
            textclass_loss = self.net_sound.compute_classification_loss(combined_pred_audio_feats, torch.cat((first_cat_idx, second_cat_idx), dim=0))
        else:
            textclass_loss = 0.0
        
        # Compute kl div loss
        audio_attn = self.net_sound.compute_audio_to_video_attn(combined_pred_audio_feats, torch.cat((feat_frames[0], feat_frames[1]), dim=0))
        text_attn = torch.cat((attn_scores[0], attn_scores[1]), dim=0)
        
        if 'framewise' in self.visual_arch:
            text_attn = text_attn.view(text_attn.size(0), audio_attn.size(1), -1)
            text_attn = text_attn.view(-1, text_attn.size(-1))
            audio_attn = audio_attn.view(-1, audio_attn.size(-1))
        kldiv_loss = F.kl_div(audio_attn, text_attn) 
        
        # Compute final loss
        final_loss = (self.visual_mask_pred_loss_weight * err) + (self.text_mask_pred_loss_weight * text_err) + (self.kl_loss_weight * kldiv_loss) + (self.textclass_loss_weight * textclass_loss)
        
        return final_loss, err, text_err, kldiv_loss, textclass_loss, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight, 'attn_scores': attn_scores, 'text_pred_masks': text_pred_masks}


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks'] 
    #pred_masks_ = outputs['text_pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]
            
def write_mp3(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3")
    
def write_wav(f, sr, x, normalized=False):
    """numpy array to WAV"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="wav")

def evaluate(netWrapper, loader, history, epoch, args, save=False):

    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    #makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # initialize HTML header
    #visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, args.num_mix+1):
        header += ['Video {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Loss weighting']
    #visualizer.add_header(header)
    vis_rows = []

    for i, batch_data in enumerate(loader):
        # forward pass
        with torch.no_grad():
            final_loss, err, text_err, kldiv_loss, textclass_loss, outputs = netWrapper.forward(batch_data, args)
        final_loss = final_loss.mean()

        loss_meter.update(final_loss.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, final_loss.item()))

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        #if len(vis_rows) < args.num_vis:
        if save and i < 20:
            output_visuals(vis_rows, batch_data, outputs, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    print('Plotting html for visualization...')
    #visualizer.add_rows(vis_rows)
    #visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args, writer):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()
    
    if 'finetune' not in args.arch_frame:
        netWrapper.module.net_frame.eval()
    else:
        for module in netWrapper.module.net_frame.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    num_steps = math.ceil(float(len(loader.dataset)) / args.batch_size_per_gpu)
    start_step = num_steps * epoch

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
    
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        final_loss, err, text_err, kldiv_loss, textclass_loss, _ = netWrapper.forward(batch_data, args)
        
        final_loss = final_loss.mean()

        # backward
        final_loss.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()
        
        err = err.mean()
        text_err = text_err.mean()
        kldiv_loss = kldiv_loss.mean()
        textclass_loss = textclass_loss.mean()
        
        curr_step = start_step + i
        writer.add_scalar('Loss/train', err, curr_step)

        # display
        if i % args.disp_iter == 0:
        
            log(args.log_path, 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_synthesizer: {}, '
                  'total_loss: {:.4f}, '
                  'mask_loss: {:.4f}, '
                  'text_mask_loss: {:.4f}, '
                  'kl_loss: {:.4f}, '
                  'textclass_loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_synthesizer,
                          final_loss.item(),
                          err.item(),
                          text_err.item(),
                          kldiv_loss.item(),
                          textclass_loss.item()))
            
            print(args.log_path, 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_synthesizer: {}, '
                  'total_loss: {:.4f}, '
                  'mask_loss: {:.4f}, '
                  'text_mask_loss: {:.4f}, '
                  'kl_loss: {:.4f}, '
                  'textclass_loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_synthesizer,
                          final_loss.item(),
                          err.item(),
                          text_err.item(),
                          kldiv_loss.item(),
                          textclass_loss.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(final_loss.item())


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
               
    lr_history = {'lr_sound': args.lr_sound, 'lr_frame': args.lr_frame}
    torch.save(lr_history, '{}/lr_history_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_frame) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    ]
                    
    if 'finetune' in args.arch_frame:
        param_groups.append({'params': net_frame.visual.parameters(), 'lr': args.lr_frame_base})
    
    if 'pre-fc' in args.arch_frame or 'post-fc' in args.arch_frame:
        param_groups.append({'params': net_frame.vis_fc.parameters(), 'lr': args.lr_frame_fc})
                    
    if 'post-fc' in args.arch_frame:
        param_groups.append({'params': net_frame.text_fc.parameters(), 'lr': args.lr_text_fc})
        
    if args.optimizer == 'adam':
        return torch.optim.Adam(param_groups)
                    
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()    
    if 'post-fc' in args.arch_frame or 'pre-fc' in args.arch_frame:
        args.num_channels = 32
    elif 'res50' in args.arch_frame:
        args.num_channels = 1024
    elif 'vitb32' in args.arch_frame:
        args.num_channels = 512
    
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound,
        args=args)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame,
        args=args)
        
    nets = (net_sound, net_frame)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    if 'solos' in args.list_val:
        dataset_train = ClipJointSOLOSMixDataset(args.list_train, args, split='train')
        dataset_val = ClipJointSOLOSMixDataset(args.list_val, args, max_sample=args.num_val, split='val')
    elif 'audioset' in args.list_val:
        dataset_train = ClipJointAudioSetMixDataset(args.list_train, args, split='train')
        dataset_val = ClipJointAudioSetMixDataset(args.list_val, args, max_sample=args.num_val, split='val')
    elif 'unseen' in args.list_val:
        dataset_train = ClipJointUnseenMixDataset(args.list_val, args, split='train')
        dataset_val = ClipJointUnseenMixDataset(args.list_val, args, max_sample=args.num_val, split='val')
    else:
        dataset_train = ClipJointMUSICMixDataset(args.list_train, args, split='train')
        dataset_val = ClipJointMUSICMixDataset(args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
        drop_last=False)
        
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit, args)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    # Eval mode
    if args.mode == 'eval' or args.mode == 'curated_eval':
        evaluate(netWrapper, loader_val, history, 0, args, save=False)
        print('Evaluation Done!')
        return
        
    history_path = os.path.join(args.ckpt, 'history_latest.pth')
    if os.path.exists(history_path):
        latest_history = torch.load(history_path)
        all_val_err = latest_history['val']['err']
        args.start_epoch = len(all_val_err) + 1
        args.best_err = min(all_val_err)
        
        latest_frame_model = torch.load(os.path.join(args.ckpt, 'frame_latest.pth'))
        latest_sound_model = torch.load(os.path.join(args.ckpt, 'sound_latest.pth'))
        
        # self.net_sound, self.net_frame, self.net_synthesizer
        netWrapper.module.net_frame.load_state_dict(latest_frame_model)
        netWrapper.module.net_sound.load_state_dict(latest_sound_model)
        
        lr_history_path = os.path.join(args.ckpt, 'lr_history_latest.pth')
        if os.path.exists(lr_history_path):
            latest_lr_history = torch.load(lr_history_path)
            args.lr_sound = latest_lr_history['lr_sound']
            args.lr_frame = latest_lr_history['lr_frame']
        history = latest_history
        
    # Set up optimizer
    optimizer = create_optimizer(nets, args)
        
    # plot losses for training
    writer = SummaryWriter(args.tensorboard_path)

    # Training loop
    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args, writer)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)
            
        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    
    if args.mode != 'eval' and args.mode != 'curated_eval':
    
        args.ckpt = args.ckpt % (args.batch_size, args.lr_sound, args.lr_frame_base, args.kl_loss_weight, args.textclass_loss_weight, args.text_mask_pred_loss_weight, args.visual_mask_pred_loss_weight)
        args.log_path = args.log_path % (args.batch_size, args.lr_sound, args.lr_frame_base, args.kl_loss_weight, args.textclass_loss_weight, args.text_mask_pred_loss_weight, args.visual_mask_pred_loss_weight)
        args.tensorboard_path = args.tensorboard_path % (args.batch_size, args.lr_sound, args.lr_frame_base, args.kl_loss_weight, args.textclass_loss_weight, args.text_mask_pred_loss_weight, args.visual_mask_pred_loss_weight)

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=False)
    elif args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
