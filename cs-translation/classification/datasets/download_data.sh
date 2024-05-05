#!/bin/bash
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
ORIGINAL_DIR="$REPO/Data/Original_Data"
PREPROCESS_DIR="$REPO/Data/Preprocess_Scripts"
PROCESSED_DIR="$REPO/Data/Processed_Data"

function download_sentiment_en_hi {
    OUTPATH=$ORIGINAL_DIR/Sentiment_EN_HI/temp
    mkdir -p $OUTPATH
    if [ ! -d $OUTPATH/SAIL_2017 ]; then
      if [ ! -f $OUTPATH/SAIL_2017.zip ]; then
        wget -c http://amitavadas.com/SAIL/Data/SAIL_2017.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/SAIL_2017.zip -d $OUTPATH
    fi

    python $PREPROCESS_DIR/preprocess_sent_en_hi.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    # rm -rf $OUTPATH 
    echo "Downloaded Sentiment EN HI"
}


download_sentiment_en_hi
