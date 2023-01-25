#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
#OPTS+="--list_train data/new_audio_video_train.csv "
#OPTS+="--list_val data/new_audio_video_val.csv "

OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "

#OPTS+="--list_train data/final_MUSIC_single_source_train.csv "
#OPTS+="--list_val data/final_MUSIC_single_source_val.csv "

#OPTS+="--ckpt ./ckpt/test_bs_%s_lr_%s_ "
#OPTS+="--log_path ./logs/test_bs_%s_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/test_bs_%s_lr_%s_ "

#OPTS+="--ckpt ./ckpt/video_res18_sop_no_dilate_maxpool_bs_%s_lr_%s_ "
#OPTS+="--log_path ./logs/video_res18_sop_no_dilate_maxpool_bs_%s_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/video_res18_sop_no_dilate_maxpool_bs_%s_lr_%s_ "

#OPTS+="--ckpt ./ckpt/frame_wise_normalize/num_frames_ablation/orig_res18_num_1_sop_maxpool_bs_%s_lr_%s_ "
#OPTS+="--log_path ./logs/frame_wise_normalize/num_frames_ablation/orig_res18_num_1_sop_maxpool_bs_%s_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/frame_wise_normalize/num_frames_ablation/orig_res18_num_1_sop_maxpool_bs_%s_lr_%s_ "

OPTS+="--ckpt ./ckpt/solos_dataset_experiments/sop_base_ablations/l1_orig_res18_num_5_sop_maxpool_bs_%s_lr_%s_model "
OPTS+="--log_path ./logs/solos_dataset_experiments/sop_base_ablations/l1_orig_res18_num_5_sop_maxpool_bs_%s_lr_%s.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/sop_base_ablations/l1_orig_res18_num_5_sop_maxpool_bs_%s_lr_%s_model "

#OPTS+="--ckpt ./ckpt/music_dataset_experiments/sop_base_ablations/l1_single_orig_res18_sop_maxpool_bs_%s_lr_%s_model "
#OPTS+="--log_path ./logs/music_dataset_experiments/sop_base_ablations/l1_single_orig_res18_sop_maxpool_bs_%s_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/music_dataset_experiments/sop_base_ablations/l1_single_orig_res18_sop_maxpool_bs_%s_lr_%s_model "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
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
#OPTS+="--num_gpus 4 "
#OPTS+="--workers 48 "
#OPTS+="--batch_size_per_gpu 20 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 5 "
OPTS+="--batch_size_per_gpu 20 "

OPTS+="--lr_frame_fc 1e-3 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u -W ignore main.py $OPTS
