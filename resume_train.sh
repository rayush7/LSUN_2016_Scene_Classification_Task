#!/usr/bin/env sh

basedir=/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_Resnet
caffepath=/media/ayush/ds-hdd/datascience/caffe

$caffepath/build/tools/caffe train --solver=$basedir/resnet152_solver.prototxt --snapshot=$basedir/finetune_resnet152_iter_14000.solverstate -gpu all 2>&1 | tee ./sample.log
