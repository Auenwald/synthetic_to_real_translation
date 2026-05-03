   ########### GTA5 -> Cityscapes, BDD - SEGFORMER B5 - AdamW #########
python train_crossattention_wrapperv3.py \
  --source_path ./synthia \
  --source_dataset_name synthiabranched \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 --lr 1.0e-5 --weight_decay 1e-3 \
  --epochs 30 --use_logging True \
  --log_file ./logs_diss/synthia_v3_wavelets_branched_ade_seed1.json \
  --skip_val_source False --decay_factor 0.995 --weight_averaging True \
  --averaging_interval 20 --train_print_steps 50 --gpu 2 \
  --mode wavelet --seed 1

  python train_crossattention_wrapperv3.py \
  --source_path ./synthia \
  --source_dataset_name synthiabranched \
  --target_paths ./cityscapes ./bdd \
  --batch_size 2 --lr 1.0e-5 --weight_decay 1e-3 \
  --epochs 30 --use_logging True \
  --log_file ./logs_diss/synthia_v3_wavelets_branched_ade_seed2.json \
  --skip_val_source False --decay_factor 0.995 --weight_averaging True \
  --averaging_interval 20 --train_print_steps 50 --gpu 2 \
  --mode wavelet --seed 2


