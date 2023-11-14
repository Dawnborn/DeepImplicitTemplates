GPU_ID=0
preprocessed_data_dir=./data
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/sofas_dit_manifoldplus --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}