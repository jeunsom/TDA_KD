import torch
import torchvision
import torchvision.transforms as transforms

from pamap2_load import get_pamap2
import numpy as np

from torch.autograd import Variable as V

NUM_WORKERS = 8


def get_PAMAP2_data(test_id=0, dataset_dir='./org_test/data/', batch_size=64):

	x_train, y_train, x_test, y_test = get_pamap2(test_id, dataset_dir)


	# subtract mean and normalize
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	# calculate per-channel means and standard deviations
	mean_image2 = np.mean(x_train, axis=(0,1,2), dtype='float64')
	std_image1 = np.std(x_train, axis=(1,2), dtype='float64') #num, h, window, ch
	std_image2 = np.mean(std_image1, axis=(0))

	print("---"*20)
	print(f'Calculated means: {mean_image2}')
	print(f'Calculated stds: {std_image2}')

	print("---"*20)

	normalize = transforms.Normalize(mean=mean_image2, std=std_image2)
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])

	train_set = []
	test_set = []

	train_label = []
	test_label = []

	train_label2 = []
	test_label2 = []

	print(x_train.shape)
	for idx, trdata in enumerate(x_train):
		trdata1 = V(simple_transform(trdata)).float().squeeze(1)
		train_set.append([trdata1, V(torch.LongTensor(y_train[idx]))[0]])


	for idx, tsdata in enumerate(x_test):
		tsdata1 = V(simple_transform(tsdata)).float().squeeze(1)
		test_set.append([tsdata1, V(torch.LongTensor(y_test[idx]))[0]])



	for idx, trlabel in enumerate(y_train):
		train_label.extend(V(torch.from_numpy(trlabel)).float().unsqueeze(0).unsqueeze(0).unsqueeze(0))
		train_label2.extend(V(torch.from_numpy(trlabel)).int().unsqueeze(0))

	for idx, tslabel in enumerate(y_test):
		test_label.extend(V(torch.from_numpy(tslabel)).float().unsqueeze(0))
		test_label2.extend(V(torch.from_numpy(tslabel)).int().unsqueeze(0))


	train_data = train_set
	test_data = test_set


	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)

	testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)

	images, labels = next(iter(trainloader))

	print('x_train.max: ', x_train.max(), x_train.min())
	print('images[0].max: ', images[0].max(), images[0].min(), images[0].mean(), images[0].std())


	print('labels[0]: ', labels[0:5])
	print('images[0].shape: ', images[0].shape)
	print('labels[0].shape: ', labels[0].shape)

	print("-----------------data loaded------------------")
	return (trainloader, testloader)



def get_PAMAP2_data4(test_id=0, datasetimg_dir='./Result_img_id/', batch_size=64):

	######load img data######
	train_id = [0,1,2,3,4,5,6,7,8]
	train_id.remove(test_id)


	x_train2 = []
	y_train2 = []
	for idn, idx in enumerate(train_id):
		if idn == 0:
			print("train_idx",idx)
			x_train1 = np.load(datasetimg_dir+"xtest_org"+str(idx)+".npy")
			y_train1 = np.load(datasetimg_dir+"ytest_org"+str(idx)+".npy")

			x_train1t = np.asarray(x_train1)
			y_train1t = np.asarray(y_train1)

			x_train2 = x_train1t
			y_train2 = y_train1t
		else:
		
			print("train_idx",idx)
			x_train1 = np.load(datasetimg_dir+"xtest_org"+str(idx)+".npy")
			y_train1 = np.load(datasetimg_dir+"ytest_org"+str(idx)+".npy")

			x_train1t = np.asarray(x_train1)
			y_train1t = np.asarray(y_train1)

			x_train2 = np.concatenate((x_train2, x_train1t), axis=0)
			y_train2 = np.concatenate((y_train2, y_train1t), axis=0)


	arr_all_xdata = np.asarray(x_train2)
	print("arr_all_xdata",arr_all_xdata.shape)

	y_train1 = np.asarray(y_train2)
	y_train = y_train1

	print("y_train2",y_train.shape)


	x_test1 = np.load(datasetimg_dir+"xtest_org"+str(test_id)+".npy") 
	y_test = np.load(datasetimg_dir+"ytest_org"+str(test_id)+".npy") 

	print("y_test",y_test.shape)

	x_train = np.transpose(arr_all_xdata, (0, 2, 3, 1))

	x_test = np.transpose(x_test1, (0, 2, 3, 1))

	# subtract mean and normalize
	x_trainimg = x_train.astype('float32')
	x_testimg = x_test.astype('float32')

	# calculate per-channel means and standard deviations
	mean_image2 = np.mean(x_trainimg, axis=(0,1,2), dtype='float64')
	std_image1 = np.std(x_trainimg, axis=(1,2), dtype='float64')
	std_image2 = np.mean(std_image1, axis=(0))

	std_image2 = np.where(std_image2==0.0, np.finfo(float).eps, std_image2)

	print("---"*20)
	print(f'Calculated means: {mean_image2}')
	print(f'Calculated stds: {std_image2}')

	print("---"*20)
	######################################################
	normalize = transforms.Normalize(mean=mean_image2, std=std_image2)
	simple_transformimg = transforms.Compose([transforms.ToTensor(), normalize])

	##load signal
	x_train, y_train, x_test, y_test = get_pamap2(test_id, '../org_test/data/')


	# subtract mean and normalize
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')


	# calculate per-channel means and standard deviations
	mean_image2 = np.mean(x_train, axis=(0,1,2), dtype='float64')
	std_image1 = np.std(x_train, axis=(1,2), dtype='float64')
	std_image2 = np.mean(std_image1, axis=(0))


	print("---"*20)
	print(f'Calculated means: {mean_image2}')
	print(f'Calculated stds: {std_image2}')


	print("---"*20)

	normalize = transforms.Normalize(mean=mean_image2, std=std_image2)
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])

	train_set = []
	test_set = []

	train_label = []
	test_label = []

	train_label2 = []
	test_label2 = []


	print(x_train.shape)
	for idx, trdata in enumerate(x_train):
		trdataimg1 = V(simple_transformimg(x_trainimg[idx])).float().squeeze(1)

		trdata1 = V(simple_transform(trdata)).float().squeeze(1)
		train_set.append([trdataimg1, trdata1, V(torch.LongTensor(y_train[idx]))[0]]) 

	for idx, tsdata in enumerate(x_test):
		tsdataimg1 = V(simple_transformimg(x_testimg[idx])).float().squeeze(1)

		tsdata1 = V(simple_transform(tsdata)).float().squeeze(1)
		test_set.append([tsdataimg1, tsdata1, V(torch.LongTensor(y_test[idx]))[0]])


	train_data = train_set
	test_data = test_set


	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)

	images, signals, labels = next(iter(trainloader))

	print('x_train.max: ', x_train.max(), x_train.min())
	print('images[0].max: ', images[0].max(), images[0].min(), images[0].mean(), images[0].std())
	print('signals[0].max: ', signals[0].max(), signals[0].min(), signals[0].mean(), signals[0].std())

	print('images[0].shape: ', images[0].shape)
	print('labels[0].shape: ', labels[0].shape)

	print("-----------------data loaded------------------")
	return (trainloader, testloader)




