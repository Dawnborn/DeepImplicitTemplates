# GPU_ID=0
preprocessed_data_dir=./data
CATEGORY=bottles
EXP=sofas_dit_manifoldplus_scanarcw
SPLITFILE=sv2_sofas_train_manifoldplus_scanarcw
# EPOCH=2000
EPOCH=2000
# /usr/wiss/lhao/anaconda3/envs/hjp_deepsdf/bin/python reconstruct_deep_implicit_templates.py -e examples/${EXP} -c ${EPOCH} --split examples/splits/${SPLITFILE}.json -d ${preprocessed_data_dir} --skip --octree
# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/wiss/lhao/anaconda3/envs/hjp_deepsdf/bin/python evaluate.py -e examples/${EXP} -c latest -s examples/splits/${TESTSPLIT}.json -d ${preprocessed_data_dir} --debug
/usr/wiss/lhao/anaconda3/envs/hjp_deepsdf/bin/python reconstruct_deep_implicit_templates_train.py