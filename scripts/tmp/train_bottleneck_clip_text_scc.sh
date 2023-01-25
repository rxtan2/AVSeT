#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=48:00:00
#$ -m bea
#$ -N nonact_text_20
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
OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_sound_pixels_bs_%s_audio_lr_%s_syn_lr_%s_frozen "
#OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_frozen.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_frozen " 

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_sound_pixels_bs_%s_audio_lr_%s_syn_lr_%s_frozen "
#OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_frozen.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_frozen " 

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_sound_pixels_bs_%s_audio_lr_%s_syn_lr_%s_finetune "
#OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_finetune.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_text_ablations/nonact_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_finetune " 

OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_sound_pixels_bs_%s_audio_lr_%s_syn_lr_%s_finetune "
OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_finetune.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_text_ablations/nonact_normalize_ratio_clip_res50_text_model_bs_%s_audio_lr_%s_syn_lr_%s_finetune " 

# Models
OPTS+="--arch_sound audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-text-normalize-finetune "

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
OPTS+="--num_gpus 1 "
OPTS+="--workers 3 "
OPTS+="--batch_size_per_gpu 20 "

OPTS+="--lr_frame_fc -1.0 "
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

python -u bottleneck_clip_text_main.py $OPTS