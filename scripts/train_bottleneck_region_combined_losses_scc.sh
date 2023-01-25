#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=48:00:00
#$ -m bea
#$ -N comb_region
#$ -j y
#$ -o output_$JOB_ID.out
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_memory=48G

module load python3/3.8.10
module load pytorch/1.9.0
module load cuda/11.1
source /projectnb/ivc-ml/rxtan/virtual_environments/object_detection/bin/activate

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "

OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bottleneck_region_distill_attn/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
OPTS+="--log_path ./logs/solos_dataset_experiments/bottleneck_region_distill_attn/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bottleneck_region_distill_attn/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

# Models
OPTS+="--arch_sound distill-audiovisual7layerunet "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-distill-attn-textproj-normalize-scale-regionmask-combined "

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
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-relate
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--num_gpus 4 "
OPTS+="--workers 1 "
OPTS+="--batch_size_per_gpu 5 "

# use latent concepts
OPTS+="--use_latent_concepts True "
OPTS+="--latent_concept_path ./precomputed_features/latent_features/solos_3_latent_concept_sgd_epoch_10000_lr_10.0_model.npy "

OPTS+="--kl_loss_weight 1e-2 "
OPTS+="--textclass_loss_weight 1e-3 "

OPTS+="--lr_frame_base -1.0 "
OPTS+="--lr_sound 3e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_clip_combined_losses_main.py $OPTS
