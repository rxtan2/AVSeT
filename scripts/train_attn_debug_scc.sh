#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=48:00:00
#$ -m bea
#$ -N debug_attn
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
OPTS+="--list_train data/single_zero_shot_seen_train.csv "
OPTS+="--list_val data/single_zero_shot_seen_val.csv "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

# OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
# OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
# OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

# OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
# OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
# OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

# OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
# OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
# OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/new_preprocessed_data/debug_unet_orig_dim_textproj_normalize_sum_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/new_preprocessed_data/debug_unet_orig_dim_textproj_normalize_sum_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/new_preprocessed_data/debug_unet_orig_dim_textproj_normalize_sum_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_audio_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_audio_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_audio_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/debug_unet_audio_trans_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/debug_unet_audio_trans_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/debug_unet_audio_trans_clip_trans_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/single_debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
OPTS+="--log_path ./logs/zero_shot_val_train_models/single_debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/single_debug_unet_audio_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/single_debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/single_debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/single_debug_unet_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

#OPTS+="--ckpt ./ckpt/zero_shot_val_train_models/single_debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "
#OPTS+="--log_path ./logs/zero_shot_val_train_models/single_debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/zero_shot_val_train_models/single_debug_unet_audio_attn_clip_attn_orig_dim_textproj_normalize_bs_%s_syn_lr_%s_ffn_dim_%s_num_layers_%s_num_heads_%s_audio_lr_%s "

# Models
OPTS+="--arch_sound debug-squeezeunet7-attn "
OPTS+="--arch_synthesizer linear "

OPTS+="--arch_frame clip-res50-joint-textproj-normalize "
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

# transformer params
OPTS+="--num_transformer_layers -1 " 
OPTS+="--num_attention_heads 2 "
OPTS+="--ffn_dim -1 "

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

python -u -W ignore debug_clip_joint_attn_main.py $OPTS
