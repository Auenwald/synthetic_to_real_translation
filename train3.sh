########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_sgd.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-3 \
  --weight_decay 1e-3 \
  --optimizer sgd \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_sgd_lr_1e03_weight_decay_1e03_ema_decay_0995_seed_0.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 3 \
  --seed 0
  ############################# 

  ########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_sgd.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-3 \
  --weight_decay 1e-3 \
  --optimizer sgd \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_sgd_lr_1e03_weight_decay_1e03_ema_decay_0995_seed_1.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 3 \
  --seed 1
  ############################# 

    ########### SYNTHIA -> Cityscapes, BDD - SEGFORMER B5 - SGD #########
python train_sgd.py \
  --source_path ./synthia \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 \
  --model_name segformer \
  --lr 1.0e-3 \
  --weight_decay 1e-3 \
  --optimizer sgd \
  --epochs 30 \
  --use_logging True \
  --log_file ./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_sgd_lr_1e03_weight_decay_1e03_ema_decay_0995_seed_2.json \
  --skip_val_source False \
  --decay_factor 0.995 \
  --weight_averaging True \
  --averaging_interval 20 \
  --train_print_steps 50 \
  --gpu 3 \
  --seed 2
  ############################# 