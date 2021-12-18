### training TTSR-ztl
python main.py --save_dir ./train/ztl/TTSR-ztl \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 9 \
               --dataset ztl \
               --dataset_dir ./dataset/ztl/ \
               --n_feats 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-3 \
               --batch_size 8 \
               --num_init_epochs 2 \
               --num_epochs 20 \
               --print_every 100 \
               --save_every 1 \
               --val_every 1

### training TTSR
# python main.py --save_dir ./train/CUFED/TTSR \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
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
#                --batch_size 8 \
#                --num_init_epochs 2 \
#                --num_epochs 50 \
#                --print_every 600 \
#                --save_every 5 \
#                --val_every 5


# ### training TTSR-rec
# python main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset False \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
#                --dataset CUFED \
#                --dataset_dir /home/v-fuyang/Data/CUFED/ \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 9 \
#                --num_init_epochs 0 \
#                --num_epochs 200 \
#                --print_every 600 \
#                --save_every 10 \
#                --val_every 10