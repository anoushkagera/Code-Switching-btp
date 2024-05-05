#!/bin/bash
BASEDIR=$1
PRETRAIN=${BASEDIR}/trimmed/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=hi_IN
TGT=en_XX
NAME=hi-en
DATADIR=${BASEDIR}/postprocessed/${NAME}
SAVEDIR=${BASEDIR}/checkpoint

fairseq-generate ${DATADIR} \
  --path $2 \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t ${TGT} -s ${SRC} \
  --bpe 'sentencepiece' --sentencepiece-model $BASEDIR/mbart.cc25/sentence.bpe.model \
  --remove-bpe 'sentencepiece' --scoring 'wer' \
  --batch-size 32 --langs $langs > output

cat output | grep -P ^H | sort -V | cut -f 3- | sed 's/\[en_XX\]//g' | python3 my_cm_tokenizer.py > output.hyp
cat ${BASEDIR}/preprocessed/test.en_XX | python3 my_cm_tokenizer.py > output.ref
cat output.hyp | sacrebleu -tok none output.ref
