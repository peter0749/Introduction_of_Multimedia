#!/bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/local/bin
TRAIN='train_loc.txt'
TEST='test_loc.txt'
rm -f $TRAIN $TEST
ls "img2hdf5/train" | sed "s/^/img2hdf5\/train\//g" - > $TRAIN
ls "img2hdf5/test" | sed "s/^/img2hdf5\/test\//g" - > $TEST
exit
