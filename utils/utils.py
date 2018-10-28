import os
import torch
from torch.autograd import Variable
from dataset import ImageDataset
from torchvision import transforms
import re
from PIL import Image
import logging

logging.basicConfig(filename="logs/common_log.log", filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


def load_available_model(checkpoint_folder):
    """Check the provided folder path for trained models"""
    if os.path.exists(checkpoint_folder):
        check = os.listdir(checkpoint_folder)  # checking if checkpoints exist to resume training
        if len(check):
            check.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                model = torch.load(os.path.join("checkpoints", check[-1]), map_location='cpu')
            else:
                model = torch.load(os.path.join("checkpoints", check[-1]))
            iter = int(re.findall(r'\d+', check[-1])[0])
            logging.info("Found model at iteration " + str(iter))
            return model, iter
    return None, 0


def get_transformation():
    """The base transformations applied before training and testing"""
    return transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_data_loaders(input_dir, batch_size=10, num_workers=6, val_size=100, test_size=100):
    """Load data from the given folder and split into train/validation/test"""
    data_transforms = get_transformation()

    train_dataset = ImageDataset(input_dir=input_dir, test_size=test_size, val_size=val_size, transform=data_transforms)
    val_dataset = ImageDataset(input_dir=input_dir, test_size=test_size, val_size=val_size, transform=data_transforms,
                               val=True)
    test_dataset = ImageDataset(input_dir=input_dir, test_size=test_size, val_size=val_size, transform=data_transforms,
                                train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_accuracy_over_set(data_loader, model, criterion, device):
    """Get Accuracy of the model on a given data set"""
    model.eval()
    total_loss = 0.0
    running_corrects = 0.0
    total_sample_size = 0.0
    for j, datapoint in enumerate(data_loader):
        datapoint['input_image'] = datapoint['input_image'].type(torch.FloatTensor)
        datapoint['label'] = datapoint['label'].type(torch.LongTensor)

        input_image = Variable(datapoint['input_image'].to(device))
        label = Variable(datapoint['label'].to(device))

        # Forward pass only to get logits/output
        outputs = model(input_image)
        _, preds = torch.max(outputs, 1)
        total_loss += criterion(outputs, label).item()
        total_sample_size += datapoint['label'].size(0)
        running_corrects += torch.sum(preds.data == label.data)

    total_loss = total_loss / total_sample_size
    return (running_corrects.item() / total_sample_size) * 100, total_loss


def image_loader(image_name):
    """load image, returns cuda tensor"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_name)
    transform = get_transformation()
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)
