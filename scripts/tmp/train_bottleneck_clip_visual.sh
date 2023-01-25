#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "
OPTS+="--ckpt ./ckpt/test_%s_base_lr_%s_fc_lr_%s "
OPTS+="--log_path ./logs/test_%s_base_lr_%s_fc_lr_%s.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_base_lr_%s_fc_lr_%s "


# Models
OPTS+="--arch_sound audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-debug-region-maxpool-textproj-randomaug "
#OPTS+="--arch_frame clip-vitb32-debug-region-meanpool-preprojection "
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
OPTS+="--num_frames 5 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 0 "
OPTS+="--batch_size_per_gpu 4 "

OPTS+="--lr_frame_base 1e-4 "
OPTS+="--lr_frame_fc 1e-3 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# optimizer
OPTS+="--optimizer sgd "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_clip_visual_main.py $OPTS
