### test
python main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./test/demo/lr/car.png \
               --ref_path ./test/demo/ref/car.png \
               --model_path ./train/CUFED/TTSR1-2/model/model_00030.pt \
               --attention_visualize True \
               --attention_roi [60:68,84:156]

# car: [120:152,84:156]
# wall: [60:68,84:156]