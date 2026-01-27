

########### SYNTHIA - SEGFORMER - SGD #########
python trainv3.py \
  --source_path ./synthia \
  --target_path ./cityscapes \
  --batch_size 4 \
  --model_name segformer \
  --lr 5.0e-3 \
  --optimizer SGD \
  --epochs 70 \
  --use_logging True \
  --log_file ./logs_final/synthia_to_cityscapes_segformer_b5_SGD_lr_5e03_decay_0995.txt \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --train_print_steps 50 \
  --gpu 0
  ############################# 

########### SYNTHIA - DEEPLAB-V3 - SGD #########
python trainv3.py \
  --source_path ./synthia \
  --target_path ./cityscapes \
  --batch_size 4 \
  --model_name deeplab \
  --lr 5.0e-3 \
  --optimizer SGD \
  --epochs 70 \
  --use_logging True \
  --log_file ./logs_final/synthia_to_cityscapes_deeplab_b5_SGD_lr_5e03_decay_0995.txt \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --train_print_steps 50 \
  --gpu 0
  ############################# 

########### SYNTHIASTYLE - DEEPLAB-V3 - SGD #########
python trainv3.py \
  --source_path ./synthiastyle \
  --target_path ./cityscapes \
  --batch_size 4 \
  --model_name deeplab \
  --lr 5.0e-3 \
  --optimizer SGD \
  --epochs 70 \
  --use_logging True \
  --log_file ./logs_final/synthiastyle_to_cityscapes_deeplab_b5_SGD_lr_5e03_decay_0995.txt \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --train_print_steps 50 \
  --gpu 0
  ############################# 


python train_crossattention_segformer.py \
  --source_path ./synthia \
  --target_path ./cityscapes \
  --batch_size 8 \
  --model_name segformer \
  --lr 1.0e-5 \
  --optimizer adamw \
  --epochs 70 \
  --use_logging True \
  --log_file ./logs_final/synthia_to_cityscapes_segformerb5_adamw_lr_1e05_decay_0995_crossAttention_RCropHFlipCJ_RandomBC_GB_fft_dct_edge_hybrid_layer.txt \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --train_print_steps 50 \
  --gpu 1

  python train_crossattention_channel_wise_gating.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_crossattention_rgb_and_edges_channel_wise_gating.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging False \
  --train_print_steps 50 \
  --gpu 1