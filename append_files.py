import pandas as pd
import numpy as np
import os

base = '/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015/train/'
#base = '/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015/val/'
#base = '/media/ayush/ds-hdd/datascience/LSUN_2016/LSUN_Dataset_2015/test/'

#output = 'lsun_val.txt'
output = 'lsun_train.txt'
#output = 'lsun_test.txt'

#For train and validation
			
def main():
 	df=pd.DataFrame(columns=['Image_Key', 'Label'])
 	count = 0

 	while(count < 10):
 		img_dir = os.path.join(base,str(count))
 		files=os.listdir(img_dir)
 		files[:] = [base+str(count)+'/'+x for x in files if x.endswith('.jpg')]
 		df_temp = pd.DataFrame(columns=['Image_Key','Label'])
 		df_temp['Image_Key']=files
 		df_temp['Label']=str(count)
 		df=df.append(df_temp)
 		count = count + 1

 	df.to_csv(base+output,sep=' ', index=False, header=False)

#def main():
#	df=pd.DataFrame(columns=['Image_Key'])
#	img_dir = os.path.join(base)
#	files=os.listdir(img_dir) 
#	files[:] = [base+x for x in files if x.endswith('.jpg')]
#	df_temp = pd.DataFrame(columns=['Image_Key'])
#	df_temp['Image_Key']=files
#	df=df.append(df_temp)
#	df.to_csv(base+output,sep=' ', index=False, header=False)




if __name__ == '__main__':
	main()		
