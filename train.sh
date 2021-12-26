### training TTSR
# python main.py --save_dir ./train/CUFED/TTSR \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 1 \
#                --dataset CUFED \
#                --dataset_dir ./dataset/CUFED/ \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 1e-2 \
#                --tpl_w 1e-2 \
#                --adv_w 1e-3 \
#                --batch_size 6 \
#                --num_init_epochs 2 \
#                --num_epochs 30 \
#                --print_every 96 \
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

### training TTSR-rec on ffhq
python main.py --save_dir ./train/CUFED/TTSR \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ffhq \
               --dataset_dir ./dataset/ffhq160/ \
               --n_feats 64 \
               --train_crop_size 40 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 0 \
               --adv_w 1e-3 \
               --batch_size 8 \
               --num_init_epochs 1 \
               --num_epochs 7 \
               --print_every 30 \
               --save_every 1 \
               --val_every 1

# profile
# python -m cProfile -o prof-report -s cumtime main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 1 \
#                --dataset ffhq \
#                --dataset_dir ./dataset/ffhq160/ \
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
#                --num_epochs 1 \
#                --print_every 50 \
#                --save_every 5 \
#                --val_every 5