### training TTSR
# python main.py --save_dir ./train/CUFED/TTSR \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 1 \
#                --dataset CUFED \
#                --dataset_dir ./dataset/CUFED/ \
#                --n_feats 64 \
#                --num_res_blocks 16+16+8+4 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 1e-2 \
#                --tpl_w 0 \
#                --adv_w 1e-3 \
#                --batch_size 6 \
#                --num_init_epochs 2 \
#                --num_epochs 30 \
#                --print_every 100 \
#                --save_every 5 \
#                --val_every 1


### training TTSR-rec
# python main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 1 \
#                --dataset CUFED \
#                --dataset_dir ./dataset/CUFED/ \
#                --n_feats 64 \
#                --train_crop_size 40 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 8 \
#                --num_init_epochs 0 \
#                --num_epochs 30 \
#                --print_every 50 \
#                --save_every 5 \
#                --val_every 1

## training TTSR on ffhq-masked
python main.py --save_dir ./train/CUFED/TTSR \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ffhq-masked \
               --dataset_dir ./dataset/ffhq256-masked/ \
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
               --num_epochs 50 \
               --print_every 96 \
               --save_every 5 \
               --val_every 1