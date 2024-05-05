#!/bin/bash

DATA_DIR="./data"
INPUT_HI_FILE="$DATA_DIR/Translate/input_hindi.txt"
OUTPUT_HINGLISH_FILE="synthetic_hinglish_sentences.txt"
MODEL_DIR='./data/mt.256.model'
 

# Run translation script
python3 translate.py \
--exp_name translate_hindi_to_hinglish \
--transformer True \
--n_enc_layers 3 \
--n_dec_layers 3 \
--share_enc 2 \
--share_dec 2 \
--share_lang_emb True \
--share_output_emb True \
--emb_dim 256 \
--langs 'en,hi' \
--mono_directions 'en,hi' \
--word_shuffle 0 \
--word_dropout 0 \
--word_blank 0 \
--pretrained_out True \
--lambda_xe_mono 1 \
--lambda_xe_otfd 1 \
--otf_num_processes 30 \
--enc_optimizer adam,lr=0.0001 \
--group_by_size True \
--batch_size 16 \
--epoch_size 200000 \
--freeze_enc_emb False \
--freeze_dec_emb False \
--reload_model $MODEL_DIR/model.pth \
--reload_enc True \
--reload_dec True \
--input_hi $INPUT_HI_FILE \
--output_hinglish $OUTPUT_HINGLISH_FILE
