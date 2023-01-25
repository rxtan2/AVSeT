#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/solos_train_5.csv "
OPTS+="--list_val data/solos_val_5.csv "

#OPTS+="--list_train data/final_MUSIC_single_source_train.csv "
#OPTS+="--list_val data/final_MUSIC_single_source_val.csv "

#OPTS+="--list_train data/final_audioset_train.csv "
#OPTS+="--list_val data/final_audioset_val.csv "

# for unimodal mask prediction
#OPTS+="--ckpt ./ckpt/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_visualmask_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_visualmask_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_visualmask_weight_%s_frozen_resnet "

OPTS+="--ckpt ./ckpt/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "
OPTS+="--log_path ./logs/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/test_%s_audio_lr_%s_visual_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "

# for all cyclic losses
#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/all_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/solos_dataset_experiments/all_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/all_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_frozen_resnet "

# for bimodal mask prediction + all cyclic losses
#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bimodal_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/solos_dataset_experiments/bimodal_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bimodal_cyclic_losses_latent/combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_frozen_resnet "

#OPTS+="--ckpt ./ckpt/solos_dataset_experiments/bimodal_cyclic_losses_latent/sample_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "
#OPTS+="--log_path ./logs/solos_dataset_experiments/bimodal_cyclic_losses_latent/sample_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet.txt "
#OPTS+="--tensorboard_path ./tensorboard_plots/solos_dataset_experiments/bimodal_cyclic_losses_latent/sample_combined_textproj_clip_res50_orig_dim_maxpool_sound_pixels_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "

# Models
#OPTS+="--arch_sound distill-audiovisual7layerunet "
OPTS+="--arch_sound distill-bimodal-audiovisual7layerunet "
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

#OPTS+="--num_gpus 1 "
#OPTS+="--workers 5   "
#OPTS+="--batch_size_per_gpu 20 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 0   "
OPTS+="--batch_size_per_gpu 4 "

# use latent concepts
OPTS+="--use_latent_concepts True "
OPTS+="--latent_concept_path ./precomputed_features/latent_features/solos_3_latent_concept_sgd_epoch_10000_lr_10.0_model.npy "
#OPTS+="--latent_concept_path ./precomputed_features/latent_features/music_multi_3_latent_concept_sgd_epoch_10000_lr_10.0_model.npy "
#OPTS+="--latent_concept_path ./precomputed_features/latent_features/audioset_full_3_latent_concept_sgd_epoch_5000_lr_10.0_model.npy "

OPTS+="--optimizer sgd "
OPTS+="--kl_loss_weight 1e-2 "
OPTS+="--textclass_loss_weight 1e-3 "
OPTS+="--text_mask_pred_loss_weight 1e-3 "
OPTS+="--visual_mask_pred_loss_weight 1.0 "

OPTS+="--lr_frame_base -1.0 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u -W ignore bottleneck_bimodal_mask_cyclic_combined.py $OPTS
#python -u -W ignore bottleneck_all_cyclic_combined.py $OPTS
#python -u -W ignore bottleneck_clip_combined_losses_main.py $OPTS
