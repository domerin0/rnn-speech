#!/bin/bash

CONFIG_FILE=config.ini
PARAMETER=signal_processing

for VALUE in 'mfcc' 'fbank'
do
    sed -i 's/\('${PARAMETER}' : \).*/\1'${VALUE}'/' ${CONFIG_FILE}
    python3 stt.py --train --tb_name ${PARAMETER}'_'${VALUE} --max_epoch 1000
    mv data/checkpoints data/checkpoint_${PARAMETER}'_'${VALUE}
    mkdir data/checkpoints
done

PARAMETER=batch_normalization

for VALUE in 'True' 'False'
do
    sed -i 's/\('${PARAMETER}' : \).*/\1'${VALUE}'/' ${CONFIG_FILE}
    python3 stt.py --train --tb_name ${PARAMETER}'_'${VALUE} --max_epoch 1000
    mv data/checkpoints data/checkpoint_${PARAMETER}'_'${VALUE}
    mkdir data/checkpoints
done

PARAMETER=dataset_size_ordering

for VALUE in 'True' 'False'
do
    sed -i 's/\('${PARAMETER}' : \).*/\1'${VALUE}'/' ${CONFIG_FILE}
    python3 stt.py --train --tb_name ${PARAMETER}'_'${VALUE} --max_epoch 1000
    mv data/checkpoints data/checkpoint_${PARAMETER}'_'${VALUE}
    mkdir data/checkpoints
done