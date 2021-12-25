### test
python main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./results/Experiment2/images/test-input-lq.png \
               --ref_path ./results/Experiment2/images/test-input-hq.png \
               --model_path ./train/CUFED/TTSR2-2/model/model_00005.pt