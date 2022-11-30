import os
import torch
import torch.nn
import torchvision
import numpy as np
import scipy.ndimage
import nibabel as nib
import SimpleITK as sitk
from models import UNet
#from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


def load_dicom(directory):
	""" перенести в класс LUNGDataset"""
	reader = sitk.ImageSeriesReader()
	dicom_names = reader.GetGDCMSeriesFileNames(directory)
	reader.SetFileNames(dicom_names)
	image_itk = reader.Execute()

	image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
	return image_zyx

def get_mask(tmp_m):
	""" перенести в класс LUNGDataset"""
	tmp_m = os.path.join(tmp_m, os.listdir(tmp_m)[0])

	mask = nib.load(tmp_m)
	mask = mask.get_fdata().transpose(2, 0, 1)
	mask = scipy.ndimage.rotate(mask, 90, (1, 2))

	return mask

img = ""
images = load_dicom(img)

tmp_m = ""
mask = get_mask(tmp_m)

class LUNGDataset(Dataset):
	def __init__(self, images, masks):
		self.images = images
		self.masks = masks
	
	def __len__(self):
		return self.images.__len__()
	
	def __getitem__(self, idx):
		image = self.images[idx]
		mask = self.masks[idx]

		image = torch.FloatTensor(image).unsqueeze(0)
		mask = torch.Tensor(mask).unsqueeze(0)

		return (image, mask)


#train
def train_one_epoch(model, optimizer, loss, dataloader):
	running_loss = 0.
	for img, msk in dataloader:
		optimizer.zero_grad()
		outputs = model(img)
		print(outputs[0].sum(axis=1))
		loss = loss_fn(outputs, msk)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		print(running_loss)

	return running_loss


dataset = LUNGDataset(images, mask)
dataloader = DataLoader(dataset, batch_size=8)
model = UNet(n_channels=1, n_classes=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.BCEWithLogitsLoss()

train_one_epoch(model, optimizer, loss_fn, dataloader)
