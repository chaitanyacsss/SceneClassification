import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from utils import *
from dataset import ImageDataset
import argparse
from PIL import Image


def main(checkpoint_folder='checkpoints/',test_sample=None):
	model_conv, iter = load_available_model(checkpoint_folder)
	if not model_conv:
		logging.info("No trained model available in the given folder")
	else:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		model_conv = model_conv.to(device)
		if not test_sample:
			_,_, test_loader = get_data_loaders(input_dir="images/", batch_size=10,num_workers=6,val_size=200,test_size=200)
			test_accuracy, _ = get_accuracy_over_set(test_loader, model_conv, nn.CrossEntropyLoss(), device)
			logging.info("Over all accuracy on the test set: "+str(test_accuracy))
		else:
			labels = ["outdoor","indoor"]
			test_image = image_loader(test_sample)
			model_conv.eval()
			outputs = model_conv(test_image)
			_, preds = torch.max(outputs, 1)
			logging.info("Predicted label is "+labels[preds.item()])
				


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Scene Recognizer with pytorch: Predict scene of an image')
	parser.add_argument('-i', '--image', default=None, type=str,help='file path to the image to be predicted (default: None/); if None, gives accuracy on the test set')
	parser.add_argument('-f', '--checkpoints', default=None, type=str,help='folder path for checkpoints (default: checkpoints/)')

	args = parser.parse_args()
	test_sample = None
	checkpoint_folder = "checkpoints"
	if args.image:
		test_sample= args.image
		logging.info("Running the model on the given image")
	else:
		logging.info("No single image given; Running the model on all test images")
	if args.checkpoints:
		checkpoint_folder= args.checkpoints
	
	main(checkpoint_folder=checkpoint_folder,test_sample=test_sample)
	
	
	
						
	

