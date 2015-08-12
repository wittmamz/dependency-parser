#!/bin/bash

DATA_TRAIN_EN="/mount/studenten/dependency-parsing/data/english/train"
DATA_DEV_EN="/mount/studenten/dependency-parsing/data/english/dev"

DATA_TRAIN_DE="/mount/studenten/dependency-parsing/data/german/train"
DATA_DEV_DE="/mount/studenten/dependency-parsing/data/german/dev"

train=1
test=1
evaluate=1

# Train the model
if [ "$train" = 1 ]; then
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -shuffle-sentences
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha -shuffle-sentences
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -shuffle-sentences -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -sentence-limit 1000 -e 5 -m model -decrease-alpha -shuffle-sentences -batch-training
    #
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -shuffle-sentences
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -decrease-alpha
    python -u tagger.py -train -i $DATA_TRAIN_DE/tiger-2.2.train.conll06 -e 10 -m model2 -decrease-alpha -shuffle-sentences
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -decrease-alpha -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -shuffle-sentences -batch-training
    #python -u tagger.py -train -i $DATA_TRAIN_EN/wsj_train.first-5k.conll06 -e 5 -m model -decrease-alpha -shuffle-sentences -batch-training
fi

# Test the model
if [ "$test" = 1 ]; then
    python -u tagger.py -test -i $DATA_DEV_DE/tiger-2.2.dev.conll06.blind -m model2 -o wsj_prediction.conll06.blind
fi

# Evaluate the results
if [ "$evaluate" = 1 ]; then
    ./eval07.pl -q -g $DATA_DEV_DE/tiger-2.2.dev.conll06.gold -s tiger-2.2.prediction.conll06.blind
    #python -u tagger.py -ev -i prediction.col -o evaluation.txt
    #python -u tagger.py -ev -i $CORPORA/test_stuff/nn.col -o evaluation.txt
    #python -u tagger.py -ev -i $CORPORA/test_stuff/leer.col -o evaluation.txt
fi


