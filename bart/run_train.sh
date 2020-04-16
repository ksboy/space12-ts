# Install newest ptl.


DATA_DIR=./space
OUTPUT_DIR=../output/space/bart
MODEL_PATH=/home/mhxia/whou/workspace/bart-large-cnn/
MAX_LENGTH=256
BATCH_SIZE=1
EVAL_BATCH_SIZE=8
NUM_EPOCHS=20
WARMUP_STEPS=1800
SEED=1
LR=3e-5

CUDA_VISIBLE_DEVICES=1,3 python run_bart_sum.py \
--model_type bart \
--model_name_or_path $MODEL_PATH \
--n_gpu 2 \
--do_train \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--fp16 \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--num_train_epochs $NUM_EPOCHS \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED \
