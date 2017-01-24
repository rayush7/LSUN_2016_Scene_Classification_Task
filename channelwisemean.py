import cv2
import pandas as pd
import numpy as np 
import sys

root='/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015'

def calculatechannelmean():
	df=pd.read_table(root+'/train/lsun_train.txt',  sep=' ', header=None)
	blue=0
	green=0
	red=0
	count = 1

	with open(root+'/train/result_mean.txt','w+') as mean_file:
		for i in df[0]:
			img=cv2.imread(i)
			img=cv2.resize(img, (224, 224))
			im=np.float32(img)
			cv2.normalize(img, im, 0, 1, cv2.cv.CV_MINMAX, cv2.cv.CV_32FC3)
			b,g,r = cv2.split(im)
			blue+=np.sum(b)
			green+=np.sum(g)
			red+=np.sum(r)
			print 'count :', count
			print 'blue sum: ', blue, 'green sum: ', green, 'red sum: ', red
			print '\n'

			if(count == 4947686):
			#if(count == 500):
				mean_file.write('\n')
				mean_file.write('count: '+str(count))
				mean_file.write('\n')
				mean_file.write('Blue Sum: ')
				mean_file.write(str(blue)) 
				mean_file.write('\n')
				mean_file.write('Green Sum: ')
				mean_file.write(str(green)) 
				mean_file.write('\n')
				mean_file.write('Red Sum: ')
				mean_file.write(str(red))
				mean_file.write('\n')

			if(count == 7000000):
			#if(count == 1000):
				mean_file.write('\n')
				mean_file.write('count: '+str(count))
				mean_file.write('\n')
				mean_file.write('Blue Sum: ')
				mean_file.write(str(blue)) 
				mean_file.write('\n')
				mean_file.write('Green Sum: ')
				mean_file.write(str(green)) 
				mean_file.write('\n')
				mean_file.write('Red Sum: ')
				mean_file.write(str(red))
				mean_file.write('\n')

			if(count == 7000000):
			#if(count == 1500):
				mean_file.write('\n')
				mean_file.write('count: '+str(count))
				mean_file.write('\n')
				mean_file.write('Blue Sum: ')
				mean_file.write(str(blue)) 
				mean_file.write('\n')
				mean_file.write('Green Sum: ')
				mean_file.write(str(green)) 
				mean_file.write('\n')
				mean_file.write('Red Sum: ')
				mean_file.write(str(red))
				mean_file.write('\n')

			if(count == 9890000):
			#if(count == 2000):
				mean_file.write('\n')
				mean_file.write('count: '+str(count))
				mean_file.write('\n')
				mean_file.write('Blue Sum: ')
				mean_file.write(str(blue)) 
				mean_file.write('\n')
				mean_file.write('Green Sum: ')
				mean_file.write(str(green)) 
				mean_file.write('\n')
				mean_file.write('Red Sum: ')
				mean_file.write(str(red))
				mean_file.write('\n')

			if(count == 9895373):
			#if(count == 3000):
				mean_file.write('\n')
				mean_file.write('count: '+str(count))
				mean_file.write('\n')
				mean_file.write('Blue Sum: ')
				mean_file.write(str(blue)) 
				mean_file.write('\n')
				mean_file.write('Green Sum: ')
				mean_file.write(str(green)) 
				mean_file.write('\n')
				mean_file.write('Red Sum: ')
				mean_file.write(str(red))
				mean_file.write('\n')

			count = count + 1 

		mean_blue=blue/(len(df[0])*224*224)
		mean_blue=mean_blue*255
		print 'mean_blue: ', mean_blue
		mean_file.write('\n')
		mean_file.write('mean_blue :' + str(mean_blue))
		mean_file.write('\n')


		mean_green=green/(len(df[0])*224*224)
		mean_green = mean_green*255
		print 'mean_green: ', mean_green
		mean_file.write('mean_green :' + str(mean_green))
		mean_file.write('\n')

		mean_red=red/(len(df[0])*224*224)
		mean_red=mean_red*255
		print 'mean_red: ', mean_red
		mean_file.write('mean_red :' + str(mean_red))
		mean_file.write('\n')

def main():
	calculatechannelmean()

if __name__ == '__main__':
	main()
