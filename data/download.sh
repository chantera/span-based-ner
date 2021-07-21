#!/bin/bash

DATA=$(cd `dirname $0` && pwd)

mkdir -p $DATA/conll2003
for split in train valid test; do
    wget https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/$split.txt -P $DATA/conll2003
    python3 $DATA/convert_conll03_to_json.py $DATA/conll2003/$split.txt $DATA/conll2003/$split.json
done
