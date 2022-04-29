train_model()
{
    MODEL_GEN=$1

    echo "Training $MODEL_GEN"
    python main.py --save_dir ./train/ztl/$MODEL_GEN \
               --reset False \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset ztl \
               --dataset_dir ./dataset/ztl/ \
               --model_gen $MODEL_GEN \
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
}

train_model ttsr-reduced
train_model ttsr-trainable-weights
train_model ttsr-soft-attention
train_model ttsr-raw