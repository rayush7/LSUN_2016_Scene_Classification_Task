# This function will predict the labels of the image.

import caffe
#import line_profiler
import pandas as pd
import numpy as np
import os
import h5py
import time
import sklearn.metrics.pairwise as skp
from PIL import Image
import sys

#path_test_image_file = '/Desktop/New/Learning_Pycaffe/test_file.txt'
path_test_image_file = '/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015/test/lsun_test.txt'
crop_size=224


#@profile
#This function should take the path of the file with Test Image Paths and predict the labels using Caffe and Deploy.prototxt
def predict_labels(path_test_image_file):
	print 'I am inside predict_labels_test'
	# Reading the text file containing the paths of the test files
	
	df=pd.read_csv(path_test_image_file, header=None, delim_whitespace=True)

	net = caffe.Net('/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_VGG19/finetune_vgg19_deploy.prototxt','/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_VGG19/finetune_vgg19_iter_12000.caffemodel',caffe.TEST)

	total_images = df[0].unique()
	num_images =len(total_images)
	count = 1
	predicted_output_list = []

	mean_values = np.array([114,126,138])

	# Predicting all the labels for the test images
	for imgs in total_images:
		#print count

		#net = caffe.Net('/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_VGG19/finetune_vgg19_deploy.prototxt','/media/ayush/ds-hdd/datascience/LSUN_2016/Finetune_VGG19/finetune_vgg_iter_12000.caffemodel',caffe.TEST)
		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))

		#transformer.set_mean('data', np.load('/home/ayush/Image_Retrieval/Retrieve_Similar_Images/mean_proto_numpy.npy').mean(1).mean(1))

		transformer.set_mean('data',mean_values)
		transformer.set_raw_scale('data', 255)
		transformer.set_channel_swap('data', (2,1,0))

		net.blobs['data'].reshape(1,3,crop_size,crop_size)   ## We are classifying just one image

		im = caffe.io.load_image(imgs)
		net.blobs['data'].data[...] = transformer.preprocess('data',im)

		out = net.forward()
		output_label = out['prob'].argmax()

		#print the results
		print 'Count :', count, 'Image :', imgs
		print 'Predicted Label :', output_label

		predicted_output_list.append(output_label)
		count = count + 1
		print '\n'

	predicted_output_nparray = np.array(predicted_output_list)
	df['Predicted_Labels']=predicted_output_nparray

	total_images=[os.path.splitext(os.path.basename(im_names))[0] for im_names in total_images]
	df[0]=total_images

	# Saving the Dataframe to csv
	df.to_csv('/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015/test/lsun_predicted_label.csv',sep = '\t',header=False,index=False)

#@profile
def main():
	caffe.set_mode_gpu()
	predict_labels(path_test_image_file)


if __name__ == "__main__":
        main()
