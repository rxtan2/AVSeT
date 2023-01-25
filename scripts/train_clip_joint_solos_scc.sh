#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=48:00:00
#$ -m bea
#$ -N res_joint_clip_20
#$ -j y
#$ -o output_$JOB_ID.out
#$ -l gpus=1
#$ -pe omp 3
#$ -l gpu_memory=48G

module load python3/3.8.10
module load pytorch/1.9.0
module load cuda/11.1
source /projectnb/ivc-ml/rxtan/virtual_environments/object_detection/bin/activate

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_1.csv "
OPTS+="--list_val data/solos_val_1.csv "

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/spatiotemporal_attn/scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_sound_pixels_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_frozen_resnet "
#OPTS+="--log_path ./logs/solos_dataset_experiments/spatiotemporal_attn/scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/spatiotemporal_attn/scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_frozen_resnet "

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/spatiotemporal_attn/finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_sound_pixels_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet "
#OPTS+="--log_path ./logs/solos_dataset_experiments/spatiotemporal_attn/finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/spatiotemporal_attn/finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet "

OPTS+="--ckpt ./ckpt/solos_dataset_experiments/spatiotemporal_attn/traintext_finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_sound_pixels_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet "
OPTS+="--log_path ./logs/solos_dataset_experiments/spatiotemporal_attn/traintext_finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/spatiotemporal_attn/traintext_finetune_scale_clip_joint_res50_orig_dim_textproj_normalize_maxpool_model_bs_%s_text_lr_%s_vis_lr_%s_audio_lr_%s_visual_lr_%s_frozen_resnet "

# Models
#OPTS+="--arch_sound squeezeunet7-upsample2 "
#OPTS+="--arch_synthesizer linear-upsample2-fc "

OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-joint-textproj-normalize-scale-maxpool-spatiotemporal-finetune-traintext "
#OPTS+="--arch_frame clip-vitb32-joint "

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

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 3 "
OPTS+="--batch_size_per_gpu 20 "

OPTS+="--lr_text_fc 1e-4 "
OPTS+="--lr_frame_fc 1e-4 "
OPTS+="--lr_frame_base 1e-4 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

# freeze visual base
OPTS+="--freeze_visual_encoder 1 "

python -u clip_joint_main.py $OPTS
