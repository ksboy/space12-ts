DATA_PATH=./cnn_dm/test.source
SUMMARIES_PATH=../output/cnn-dm/bart
MODEL_PATH=/home/mhxia/whou/workspace/bart-large-cnn

CUDA_VISIBLE_DEVICES=2 python evaluate_cnn.py \
    --model_name $MODEL_PATH \
    --source_path $DATA_PATH \
    --output_path $SUMMARIES_PATH \
    --device cuda \
    --bs 8