#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_1.csv "
OPTS+="--list_val data/solos_val_1.csv "
#OPTS+="--ckpt ./ckpt/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s "

OPTS+="--ckpt ./ckpt/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s_visual_lr_%s "
OPTS+="--log_path ./logs/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s_visual_lr_%s.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_base_lr_%s_fc_lr_%s_audio_lr_%s_visual_lr_%s "

# Models
#OPTS+="--arch_sound squeezeunet7-upsample2 "
#OPTS+="--arch_synthesizer linear-upsample2-fc "

OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "

#OPTS+="--arch_frame clip-res50-joint-orig-dim-top "
OPTS+="--arch_frame clip-res50-joint-textproj-normalize-scale-finetune-traintext-maxpool-framewise "

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
OPTS+="--num_frames 5 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-relate
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 0 "
OPTS+="--batch_size_per_gpu 4 "

OPTS+="--lr_text_fc 1e-3 "
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

python -u -W ignore clip_joint_main.py $OPTS
