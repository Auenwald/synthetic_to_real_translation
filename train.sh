########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_crossattention_wrapperv2.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0_wrapperv2_dct.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 0
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_crossattention_wrapperv2.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_1_wrapperv2_dct.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 1
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_crossattention_wrapperv2.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_2_wrapperv2_dct.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 2
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_crossattention_wrapperv2.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_3_wrapperv2_dct.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 3
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_crossattention_wrapperv2.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4_wrapperv2_dct.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 4
  ############################# 