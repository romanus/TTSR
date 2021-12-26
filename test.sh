### test
python main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./results/Experiment2/images/train-input-lq-384.png \
               --ref_path ./results/Experiment2/images/train-ref-hq-384.png \
               --model_path ./train/CUFED/TTSR2-4/model/model_00005.pt