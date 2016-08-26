#!/bin/bash

CONFIG_FILE=config.ini
PARAMETER=dropout

for VALUE in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'
do
    sed -i 's/\('${PARAMETER}' : \).*/\1'${VALUE}'/' ${CONFIG_FILE}
    python3 stt.py --train --tb_name ${PARAMETER}'_'${VALUE} --max_epoch 20000
done