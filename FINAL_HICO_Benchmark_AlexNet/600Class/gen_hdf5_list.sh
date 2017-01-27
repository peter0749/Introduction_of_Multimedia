#!/bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/local/bin
TRAIN='train_loc.txt'
TEST='test_loc.txt'
TRAINL='train_label'
TESTL='test_label'
LDIR='label_list'
rm -f $TRAIN $TEST
rm -rf $LDIR
mkdir $LDIR
ls "img2hdf5/train" | sed "s/^/img2hdf5\/train\//g" - > $TRAIN
ls "img2hdf5/test" | sed "s/^/img2hdf5\/test\//g" - > $TEST
for i in {1..600}
do
    echo -e "img2hdf5/Labels/Test/Class-$i.hdf5" > "$LDIR/$TESTL-$i.txt"
    echo -e "img2hdf5/Labels/Train/Class-$i.hdf5" > "$LDIR/$TRAINL-$i.txt"
done
exit
