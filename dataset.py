import numpy as np
import scipy.ndimage
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader


def _load_dicom(directory):
	reader = sitk.ImageSeriesReader()
	dicom_names = reader.GetGDCMSeriesFileNames(directory)
	reader.SetFileNames(dicom_names)
	image_itk = reader.Execute()

	image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)

	return image_zyx

def _get_mask(tmp_m):
	tmp_m = os.path.join(tmp_m, os.listdir(tmp_m)[0])

	mask = nib.load(tmp_m)
	mask = mask.get_fdata().transpose(2, 0, 1)
	mask = scipy.ndimage.rotate(mask, 90, (1, 2))

	return mask


class LUNGDataset(Dataset):
	def __init__(self, images, masks):
		self.images = imagimageses
		self.masks = masks
	
	def __len__(self):
		return self.images.__len__()
	
	def __getitem__(self, idx):
		image = self.images[idx]
		mask = self.masks[idx]

		image = torch.FloatTensor(image).unsqueeze(0)
		mask = torch.Tensor(mask).unsqueeze(0)

		return (image, mask)

images = _load_dicom(DIRECTORY)
masks = _get_mask(TMP_M)

dmataset = LUNGDataset(images, masks)
