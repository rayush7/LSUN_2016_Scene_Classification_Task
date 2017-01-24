import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from pylab import *


def train_resnet():
	#solver = caffe.get_solver('/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_Resnet/resnet152_solver.prototxt')
	#tmp = sys.stdout
	#sys.stdout = open('./resnet_loss.log', 'wt')
	#print 'Started training'
	#solver.solve()
	#sys.stdout.close()
	#sys.stdout = tmp

	solver = caffe.SGDSolver('/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_Resnet/resnet152_solver.prototxt')
	Niter = 10
	Display_iter = 1
	Test_iter = 1500
	Test_interval = 100

	# train loss
	Train_loss = zeros(Niter, ceil (1/ Display_iter))
	# test loss
	Test_loss = zeros(Niter, ceil(1/ Test_interval))
	# test accuracy
	Test_acc = zeros(Niter, ceil(1/ Test_interval))

	# iteration 0, are not included in the
	solver.step(1)

	# auxiliary variable
	_train_loss =0
	_test_loss =0
	_accuracy=0

	# solution
	for It in range(Niter):
		# a solution
		solver.step (1)
		# calculation of train loss
		_train_loss = solver.net.blobs['SoftmaxWithloss1'].data


		if It%Display_iter==0:
        		# average train loss
			Train_loss[It/Display_iter] = _train_loss / Display_iter
			_train_loss = 0 

		if It%Test_interval==0:
			for Test_it in range(Test_iter):
				# a test
				solver.test_nets[0].forward()
				# calculation of test loss
				_test_loss = solver.test_nets[0].blobs['SoftmaxWithloss1'].data
				# calculation of test accuracy
				_accuracy = solver.test_nets[0].blobs['accuracy'].data
				# average test loss
				Test_loss[It/Test_interval] =_test_loss / test_iter
				# average test accuracy
				Test_acc[It/Test_interval] =_accuracy/ test_iter
				_test_loss = 0
				_accuracy = 0


#	Train loss, test # draw loss and accuracy curve
#	PrintThe train loss and test'\nplot accuracy\n'

	#Print the Loss
	print 'Train Loss :',Train_loss
	print 'Test Loss :', Test_loss


	# Plot the accuracy
	#_ (ax1 = plt.subplots).
	_, ax1 = plt.subplots(figsize=(15, 10))

	aX2 = ax1.twinx ()

	# train loss - > Green
	ax1.plot (display_iter * arange (Len(train_loss)), train_loss,'g')
	# test loss - > yellow
	ax1.plot (test_interval * arange (Len(test_loss)), test_loss,'y')
	# test accuracy - > Red
	ax2.plot (test_interval * arange (Len(test_acc)), test_acc,'r')

	ax1.set_xlabel ('iteration')
	ax1.set_ylabel ('loss')
	ax2.set_ylabel ('accuracy')
	Plt.show ()


def main():
	train_resnet()



if __name__=='__main__':
	caffe.set_device(0)
	caffe.set_mode_gpu()
	main()
