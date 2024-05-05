#!/bin/bash

if [ -d "/home/bhavya24/delete/temp_dir/" ]; then
    echo "Deleting base directory"
    rm -rf /home/bhavya24/delete/temp_dir/
fi

mkdir /home/bhavya24/delete/temp_dir/
mkdir /home/bhavya24/delete/temp_dir/preprocessed
mkdir /home/bhavya24/delete/temp_dir/postprocessed

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
TRAIN=train
VALID=valid
TEST=test
EXPERIMENT_ID=$1
SPM=spm_encode
DATA=/home/bhavya24/delete/temp_dir/preprocessed
DATADIR=/home/bhavya24/delete/temp_dir/postprocessed
SRC=hi_IN
TGT=en_XX
NAME=hi-en
BASEDIR=/scratch/bhavya24/${EXPERIMENT_ID}/
MODEL=${BASEDIR}/mbart.cc25/sentence.bpe.model
DICT=${BASEDIR}/trimmed/dict.txt
INPUT_FILE=$3
OUTPUT_FILE=$4

for Item in 'train' 'valid' 'test';
	do
		${SPM} --model=${MODEL} < ${INPUT_FILE} > ${DATA}/${Item}.spm.${SRC}
		${SPM} --model=${MODEL} < ${INPUT_FILE} > ${DATA}/${Item}.spm.${TGT}
	done

fairseq-preprocess \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir $DATADIR \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70

fairseq-generate ${DATADIR} \
  --path $2 \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t ${TGT} -s ${SRC} \
  --bpe 'sentencepiece' --sentencepiece-model $BASEDIR/mbart.cc25/sentence.bpe.model \
  --remove-bpe 'sentencepiece' --scoring 'wer' \
  --batch-size 16 --langs $langs > /home/bhavya24/delete/temp_dir/output

cat /home/bhavya24/delete/temp_dir/output | grep -P ^H | sort -V | cut -f 3- | sed 's/\[en_XX\]//g' > ${OUTPUT_FILE}
