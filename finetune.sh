#!/usr/bin/env sh

basedir=/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_Resnet
caffepath=/media/ayush/ds-hdd/datascience/caffe

$caffepath/build/tools/caffe train --solver=$basedir/resnet152_solver.prototxt -weights $basedir/ResNet-152-model.caffemodel -gpu all 2>&1 | tee ./resnet152_loss_finetune_attempt6.log
