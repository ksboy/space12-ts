DATA_PATH=../../cnn/stories/
# DATA_PATH=/home/mhxia/whou/workspace/code_repo/space12-ner-allennlp/data/AIAA_with_text.csv
SUMMARIES_PATH=../../output/cnn-dm/bert-abs/
MODEL_PATH=/home/mhxia/whou/workspace/bert-abs

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name $MODEL_PATH \
    --documents_dir $DATA_PATH \
    --summaries_output_dir $SUMMARIES_PATH \
    --no_cuda false \
    --batch_size 80 \
    --min_length 50 \
    --max_length 200 \
    --beam_size 5 \
    --alpha 0.95 \
    --block_trigram true \
    --compute_rouge true ;

#       min max ave
# space  1 33 10
# cnn-dm 0 515 129