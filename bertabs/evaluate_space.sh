# DATA_PATH=../cnn/stories/
DATA_PATH=../data/AIAA_with_text.csv
SUMMARIES_PATH=../output/space/bert-abs/space.predict
MODEL_PATH=/home/mhxia/whou/workspace/pretrained_models/bert-abs/

CUDA_VISIBLE_DEVICES=3 python run_summarization.py \
    --model_name $MODEL_PATH \
    --documents_path $DATA_PATH \
    --summaries_output_path $SUMMARIES_PATH \
    --no_cuda false \
    --batch_size 8 \
    --min_length 5 \
    --max_length 35 \
    --beam_size 5 \
    --alpha 0.95 \
    --block_trigram true \
    --compute_rouge true ;

#       min max ave
# space  1 33 10
# cnn-dm 0 515 129