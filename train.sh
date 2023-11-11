GPU_ID=0
preprocessed_data_dir=./data
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/bottles_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}