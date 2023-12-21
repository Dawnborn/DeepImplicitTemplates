GPU_ID=0 
preprocessed_data_dir=./data
CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/wiss/lhao/anaconda3/envs/hjp_deepsdf/bin/python train_deep_implicit_templates.py -e examples/sofas_dit_manifoldplus_scanarcw --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}