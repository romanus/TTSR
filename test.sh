### test
python main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./results/Experiment2/images/input-lq.png \
               --ref_path ./results/Experiment2/images/ref-hq.png \
               --model_path ./train/CUFED/TTSR2-3/model/model_00007.pt