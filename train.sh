

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

