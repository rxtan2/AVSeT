OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train data/music_train.csv "
OPTS+="--list_val data/music_val.csv "

OPTS+="--ckpt ./ckpt/model_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "
OPTS+="--log_path ./logs/model_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet.txt "
OPTS+="--tensorboard_path ./tensorboard_plots/model_bs_%s_audio_lr_%s_vis_lr_%s_kldiv_weight_%s_textclass_weight_%s_textmask_weight_%s_visualmask_weight_%s_frozen_resnet "

OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame clip-res50-distill-attn-textproj-normalize-scale-maxpool-framewise "

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

OPTS+="--num_gpus 8 "
OPTS+="--workers 2 "
OPTS+="--batch_size_per_gpu 4 "

# use latent concepts
OPTS+="--use_latent_concepts True "
OPTS+="--latent_concept_path /path/to/extracted latent embeddings "

OPTS+="--kl_loss_weight 1e-3 "
OPTS+="--textclass_loss_weight 1e-3 "
OPTS+="--text_mask_pred_loss_weight 1e-3 "
OPTS+="--visual_mask_pred_loss_weight 1.0 "

OPTS+="--optimizer sgd "
OPTS+="--lr_frame_base -1.0 "
OPTS+="--lr_sound 5e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

export OMP_NUM_THREADS=1

python -u -W ignore bottleneck_bimodal_mask_cyclic_combined.py $OPTS
