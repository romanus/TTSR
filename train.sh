echo "Training ttsr-reduced"
python main.py --save_dir ./train/ztl-masked/ttsr-reduced \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ztl-masked \
               --dataset_dir ./dataset/ztl256-masked/ \
               --model_gen ttsr-reduced \
               --n_feats 64 \
               --train_crop_size 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 0 \
               --adv_w 1e-3 \
               --batch_size 3 \
               --num_init_epochs 2 \
               --num_epochs 4 \
               --print_every 42 \
               --save_every 1 \
               --val_every 1

echo "Training ttsr-trainable-weights"
python main.py --save_dir ./train/ztl-masked/ttsr-trainable-weights \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ztl-masked \
               --dataset_dir ./dataset/ztl256-masked/ \
               --model_gen ttsr-trainable-weights \
               --n_feats 64 \
               --train_crop_size 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 0 \
               --adv_w 1e-3 \
               --batch_size 3 \
               --num_init_epochs 2 \
               --num_epochs 4 \
               --print_every 42 \
               --save_every 1 \
               --val_every 1

echo "Training ttsr-soft-attention"
python main.py --save_dir ./train/ztl-masked/ttsr-soft-attention \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ztl-masked \
               --dataset_dir ./dataset/ztl256-masked/ \
               --model_gen ttsr-soft-attention \
               --n_feats 64 \
               --train_crop_size 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 0 \
               --adv_w 1e-3 \
               --batch_size 3 \
               --num_init_epochs 2 \
               --num_epochs 4 \
               --print_every 42 \
               --save_every 1 \
               --val_every 1

echo "Training ttsr-raw"
python main.py --save_dir ./train/ztl-masked/ttsr-raw \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ztl-masked \
               --dataset_dir ./dataset/ztl256-masked/ \
               --model_gen ttsr-raw \
               --n_feats 64 \
               --train_crop_size 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 0 \
               --adv_w 1e-3 \
               --batch_size 3 \
               --num_init_epochs 2 \
               --num_epochs 4 \
               --print_every 42 \
               --save_every 1 \
               --val_every 1