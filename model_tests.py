import unittest

import torch.optim as optim
from torch import nn

from utils import *


class TestModel(unittest.TestCase):
    model, iter = load_available_model("checkpoints/")

    def test_load_model(self):
        """check if model is loaded properly; current model is at 4000 iter"""
        self.assertTrue(self.__class__.model)
        self.assertEqual(self.iter, 4000)

    def test_dataloader(self):
        """Checks if data loader is loading correctly"""
        train, val, test = get_data_loaders("images/", batch_size=10, num_workers=6, val_size=200, test_size=200)
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertEqual(train.batch_size, val.batch_size)
        self.assertEqual(val.batch_size, test.batch_size)
        self.assertEqual(val.batch_size, 10)
        self.assertEqual(val.dataset.__len__(), 200)
        self.assertEqual(test.dataset.__len__(), 200)

    def test_model_sanity(self):
        """Runs a training step and checks if weights have been updated"""

        data_loader, _, _ = get_data_loaders("images/", num_workers=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer = optim.SGD(self.__class__.model.fc.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.__class__.model.train()
        for i, datapoint in enumerate(data_loader):
            datapoint['input_image'] = datapoint['input_image'].type(torch.FloatTensor)
            datapoint['label'] = datapoint['label'].type(torch.LongTensor)

            input_image = Variable(datapoint['input_image'].to(device))
            label = Variable(datapoint['label'].to(device))

            input_image = Variable(input_image.to(device))
            label = Variable(label.to(device))

            pre_weights = [self.__class__.model.parameters()]
            optimizer.zero_grad()
            outputs = self.__class__.model(input_image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            post_weights = [self.__class__.model.parameters()]
            self.assertTrue(post_weights != pre_weights)
            break


if __name__ == '__main__':
    unittest.main()
