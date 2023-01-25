#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "

OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "

#OPTS+="--ckpt ./ckpt/test_bs_%s_lr_%s_ "
#OPTS+="--log_path ./logs/test_bs_%s_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/test_bs_%s_lr_%s_ "

OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_sop_runs/audio_3e-3_ratio_num_3_dim_512_video_res18_orig_sop_sound_pixels_bs_%s_lr_%s_frozen_resnet "
OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_sop_runs/audio_3e-3_ratio_num_3_dim_512_video_res18_orig_sop_sound_pixels_bs_%s_lr_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_sop_runs/audio_3e-3_ratio_num_3_dim_512_video_res18_orig_sop_sound_pixels_bs_%s_lr_%s_frozen_resnet "

# Models

# debugging new vocoding process
OPTS+="--arch_sound audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 1024 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 2 "
OPTS+="--workers 5 "
OPTS+="--batch_size_per_gpu 10 "

#OPTS+="--num_gpus 2 "
#OPTS+="--workers 0 "
#OPTS+="--batch_size_per_gpu 4 "

OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 3e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_sop_main.py $OPTS
