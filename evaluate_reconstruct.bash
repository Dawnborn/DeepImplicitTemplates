GPU_ID=0
preprocessed_data_dir=./data
CATEGORY=bottles
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_deep_implicit_templates.py -e examples/${CATEGORY}_dit -c latest --split examples/splits/sv2_${CATEGORY}_test.json -d ${preprocessed_data_dir} --skip --octree
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -e examples/${CATEGORY}_dit -c 2000 -s examples/splits/sv2_${CATEGORY}_test.json -d ${preprocessed_data_dir} --debug