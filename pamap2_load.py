import os
import scipy.io
import numpy
import random

import numpy as np
import pickle

NUM_WORKERS = 2


def get_pamap2(test_id=0, dataset_dir='data/'):

	folder_root = dataset_dir

	data = folder_root + "pamap2_12cls9sbj.data"

	fpamap = pickle.load( open( data, "rb" ) , encoding="latin1")

	cls_id = [24,1,2,3,4,5,6,7,12,13,16,17]
	train_id = [0,1,2,3,4,5,6,7,8]
	train_id.remove(test_id)

	print("train_id :", train_id, "test_id :", test_id)

	train_temp = 0
	test_temp = int(np.sum(len(fpamap[test_id][1])))
	for i in range(8):
		train_temp += len(fpamap[i][1])
	train_temp = train_temp - test_temp

	window_length = 100
	step_size = 22


	tempx = []
	tempy = []

	tempx2 = []
	tempy2 = []

	count = 0
	count2 = 0


	print('-----------------train subject data---------------------')

	
	for _, i in enumerate(train_id):
		print(i, len(fpamap[i][1]))
		for j in range(22,len(fpamap[i][1])-100,step_size):
			idx = fpamap[i][1][j]
			if (idx >= 0 and idx <= 11) and idx == fpamap[i][1][j+99]:
				tempx.append(fpamap[i][0][j:j+100])
				tempy.append(fpamap[i][1][j])

				count += 1
	
	print('-----------------test subject data---------------------')
	i = test_id
	if i < 9:
		print(i, len(fpamap[i][1]))
		for j in range(22,len(fpamap[i][1])-100,step_size):
			idx = fpamap[i][1][j]
			if (idx >= 0 and idx <= 11) and idx == fpamap[i][1][j+99]:
				tempx2.append(fpamap[i][0][j:j+100])
				tempy2.append(fpamap[i][1][j])

				count2 += 1	

	print(len(tempx),len(tempx2), train_temp, test_temp)



	x_train = np.zeros((count,1,window_length,40))
	x_test = np.zeros((count2,1,window_length,40))
	y_train = np.zeros((count,1))
	y_test = np.zeros((count2,1))

	for idx in range(len(tempy)):

		x_train[idx,0,:,:] = tempx[idx][:][:]
		y_train[idx] = tempy[idx]


	for idx in range(len(tempy2)):
		x_test[idx,0,:,:] = tempx2[idx][:][:]
		y_test[idx] = tempy2[idx]


	print(f'train: {len(x_train)}   test: {len(x_test)}   class: {len(cls_id)}')

	print('-------------Train data #-----------------')
	for idx in range(len(cls_id)):
		print(idx, "-", (y_train == idx).sum())

	print('-------------Test data #-----------------')
	for idx in range(len(cls_id)):
		print(idx, "-", (y_test == idx).sum())

	return x_train, y_train, x_test, y_test













	# Re-save as Python 3 pickle
	#with open(new_pkl, "wb") as outfile:
	#	pickle.dump(fpamap, outfile)




	#x_train1 = np.zeros((train_num,1,window_length,40)) #num, hei, wid, ch
	#x_test1 = np.zeros((test_num,1,window_length,40))

	#y_train1 = np.zeros((train_num,1))
	#y_test1 = np.zeros((test_num,1))






if __name__ == "__main__":

	print("get PAMAP2")
	print(get_pamap2(0))


	#print("get_gene")
	#print(get_gene(16))

	#print("get_gene")
	#print(get_gene(100))

	print("---"*20)
	print("---"*20)


