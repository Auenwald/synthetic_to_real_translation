########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train.py \
  --source_path ./gta5 \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0_new.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 1 \
  --seed 0
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_09_averaging_interval_20_seed_1_new.json \
  --skip_val_source False \
  --decay_factor 0.9 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 1
  ############################# 

########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_09_averaging_interval_20_seed_2_new.json \
  --skip_val_source False \
  --decay_factor 0.9 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 2
  ############################# 

  ########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_09_averaging_interval_20_seed_3_new.json \
  --skip_val_source False \
  --decay_factor 0.9 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 3
  ############################# 

    ########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-5 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_09_averaging_interval_20_seed_4_new.json \
  --skip_val_source False \
  --decay_factor 0.9 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 0 \
  --seed 4
  ############################# 