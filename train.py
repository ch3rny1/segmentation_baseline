import os
import torch
import torch.nn
import torchvision
from models import UNet
from dataset import dataset
from config import EPOCH_NUM, LOSS_FN, OPTIMIZER, MODEL


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


if __name__ == "__main__":

	dataloader = DataLoader(dataset, batch_size=8)

	for _ in range(EPOCH_NUM):
		train_one_epoch(MODEL, OPTIMIZER, LOSS_FN, dataloader)
