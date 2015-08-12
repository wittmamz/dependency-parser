#!/bin/bash

DATA_TRAIN_EN="/mount/studenten/dependency-parsing/data/english/train"
DATA_DEV_EN="/mount/studenten/dependency-parsing/data/english/dev"

DATA_TRAIN_DE="/mount/studenten/dependency-parsing/data/german/train"
DATA_DEV_DE="/mount/studenten/dependency-parsing/data/german/dev"

train=0
test=1
evaluate=1

# Train the model
if [ "$train" = 1 ]; then
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -shuffle-sentences
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha
    python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 100 -e 5 -m model_en -decrease-alpha -shuffle-sentences
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -shuffle-sentences -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha -shuffle-sentences -batch-training
    #
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -shuffle-sentences
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -decrease-alpha
    #python -u dependency-parser.py -train -i $DATA_TRAIN_DE/tiger-2.2.train.conll06 -e 10 -m model_de -decrease-alpha -shuffle-sentences
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -decrease-alpha -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -shuffle-sentences -batch-training
    #python -u dependency-parser.py -train -i $DATA_TRAIN_EN/wsj_train.conll06 -e 5 -m model -decrease-alpha -shuffle-sentences -batch-training
fi

# Test the model
if [ "$test" = 1 ]; then
    #python -u dependency-parser.py -test -i $DATA_DEV_EN/tiger-2.2.dev.conll06.blind -m model_de -o tiger-2.2.prediction.conll06.blind
    python -u dependency-parser.py -test -i $DATA_DEV_EN/wsj_dev.conll06.blind -m model_en -o wsj_prediction.conll06.blind
fi

# Evaluate the results
if [ "$evaluate" = 1 ]; then
    #./eval07.pl -q -g $DATA_DEV_DE/tiger-2.2.dev.conll06.gold -s tiger-2.2.prediction.conll06.blind
	./eval07.pl -q -g $DATA_DEV_EN/wsj_dev.conll06.gold -s wsj_prediction.conll06.blind
fi


