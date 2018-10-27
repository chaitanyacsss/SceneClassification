import random
import os
from torch.utils.data.dataset import Dataset
from PIL import Image



class ImageDataset(Dataset):

    def __init__(self, input_dir, val_size=100, test_size=100, train=True, val=False, transform=None):
        self.input_dir=input_dir
        self.transform=transform
        if val:
            train = False
        self.train=train
        self.val=val
        self.val_size = val_size
        self.test_size = test_size
        self.files = []
        self.labels = ["outdoor","indoor"]
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            for filename in filenames:
                self.files.append(dirpath+"/"+filename)
                
        random.Random(4).shuffle(self.files)


    def __len__ (self):
        if self.train:
            return len(self.files) - self.test_size - self.val_size
        elif self.val:
            return self.val_size
        else:
            return self.test_size

    def __getitem__(self,idx):
        if self.train:
            idx = idx
        elif self.val:
            idx = idx + len(self.files) - self.val_size - self.test_size
        else:
            idx = idx + len(self.files) - self.test_size
            
        input_image=Image.open(self.files[idx])
        label = self.labels.index(self.files[idx].split("/")[-2])
        
        if self.transform:
            input_image = self.transform(input_image)

        sample = {'input_image': input_image, 'label': label}             

        return sample