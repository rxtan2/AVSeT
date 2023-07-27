#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_1.csv "
OPTS+="--list_val data/solos_val_1.csv "
#OPTS+="--ckpt ./ckpt/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s "

OPTS+="--ckpt ./ckpt/test_%s_audio_lr_%s_visual_lr_%s_regloss_lr_%s_latent_lr_%s "
OPTS+="--log_path ./logs/test_%s_audio_lr_%s_visual_lr_%s_regloss_lr_%s_latent_lr_%s.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_audio_lr_%s_visual_lr_%s_regloss_lr_%s_latent_lr_%s "

# Models
#OPTS+="--arch_sound squeezeunet7-upsample2 "
#OPTS+="--arch_synthesizer linear-upsample2-fc "

OPTS+="--arch_sound ft-latent-distill-audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "

#OPTS+="--arch_frame clip-res50-joint-orig-dim-top "
OPTS+="--arch_frame clip-res50-distill-attn-textproj-normalize-scale-kldiv-framewise "

OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-relate
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 0 "
OPTS+="--batch_size_per_gpu 4 "

# use latent concepts
OPTS+="--use_latent_concepts True "
OPTS+="--latent_concept_path ./precomputed_features/latent_features/visual_frames_latent_concepts.npy "

OPTS+="--reg_loss_weight 1e-2 "


OPTS+="--lr_latent 1e-4 "
OPTS+="--lr_frame_base 1e-4 "
OPTS+="--lr_frame_fc 1e-3 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_clip_ft_latent_distill.sh $OPTS