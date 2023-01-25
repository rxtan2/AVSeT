#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=48:00:00
#$ -m bea
#$ -N audioset_comb
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
OPTS+="--list_train data/final_audioset_train.csv "
OPTS+="--list_val data/final_audioset_val.csv "

#OPTS+="--ckpt ./ckpt/audioset_dataset_experiments/bottleneck_clip_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/audioset_dataset_experiments/bottleneck_clip_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/audioset_dataset_experiments/bottleneck_clip_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

OPTS+="--ckpt ./ckpt/audioset_dataset_experiments/bottleneck_clip_latent/dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
OPTS+="--log_path ./logs/audioset_dataset_experiments/bottleneck_clip_latent/dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/audioset_dataset_experiments/bottleneck_clip_latent/dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

#OPTS+="--ckpt ./ckpt/audioset_dataset_experiments/bottleneck_clip_latent/randomaug_dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/audioset_dataset_experiments/bottleneck_clip_latent/randomaug_dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/audioset_dataset_experiments/bottleneck_clip_latent/randomaug_dup_5_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

#OPTS+="--ckpt ./ckpt/audioset_dataset_experiments/bottleneck_clip_latent/adam_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/audioset_dataset_experiments/bottleneck_clip_latent/adam_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/audioset_dataset_experiments/bottleneck_clip_latent/adam_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

# Models
OPTS+="--arch_sound distill-audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-distill-attn-textproj-normalize-scale-maxpool-framewise-combined "

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

# audio-relate
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 3 "
OPTS+="--batch_size_per_gpu 20 "

# use latent concepts
OPTS+="--use_latent_concepts True "
OPTS+="--latent_concept_path ./precomputed_features/latent_features/audioset_full_3_latent_concept_sgd_epoch_5000_lr_10.0_model.npy "

#OPTS+="--kl_loss_weight 1e-2 "
#OPTS+="--textclass_loss_weight 1e-3 "

OPTS+="--kl_loss_weight 1e-4 "
OPTS+="--textclass_loss_weight 1e-4 "

OPTS+="--optimizer sgd "
OPTS+="--lr_frame_base -1.0 "
OPTS+="--lr_sound 4e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_clip_combined_losses_main.py $OPTS
