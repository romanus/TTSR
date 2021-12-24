### test
python main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./results/Experiment1/images/input-lq.png \
               --ref_path ./results/Experiment1/images/ref-hq.png \
               --model_path ./train/CUFED/TTSR1-3/model/model_00030.pt