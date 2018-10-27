import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from utils import *
from dataset import ImageDataset
import argparse
from trainer import train_model


def main(checkpoint_folder='checkpoints/'):

	model_conv, iter = load_available_model(checkpoint_folder)

	if not model_conv:
		logging.info("No saved model found; starting train with a pretrained resnet.")
		model_conv = models.resnet152(pretrained=True)
		for param in model_conv.parameters():
			param.requires_grad = False
		num_ftrs = model_conv.fc.in_features
		model_conv.fc = nn.Linear(num_ftrs, 2)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model_conv = model_conv.to(device)
	
	criterion = nn.CrossEntropyLoss()

	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
	
	train_loader, val_loader, _ = get_data_loaders(input_dir="images", batch_size=10,num_workers=6,val_size=200,test_size=200)
	logging.info("starting train")
	model_conv = train_model(model_conv, train_loader, val_loader, criterion, optimizer_conv, exp_lr_scheduler, checkpoint_folder, device, num_epochs=25)	


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Scene Recognizer with pytorch - run train')
	parser.add_argument('-r', '--resume', default="checkpoints", type=str,help='folder path to latest checkpoint (default: checkpoints/)')

	args = parser.parse_args()
	if args.resume:
		checkpoint_folder= args.resume
	
	main(checkpoint_folder)
	
	
	
						
	

