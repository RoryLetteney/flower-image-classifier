import argparse
import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np

parser = argparse.ArgumentParser(description="Train model of choice on data of choice.")
parser.add_argument('data_dir', type=str, help='Path to data directory from root.')
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, nargs='+', default=[256, 128])
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', type=bool, default=False)

args = parser.parse_args()


class Model:
    def __init__(self, data_dir='./flowers', arch='vgg16', save_dir='.', learning_rate=0.001, hidden_units=[256, 128], epochs=5, gpu=False):
        self.data_dir = data_dir if data_dir != '' else './flowers'
        self.arch = arch
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu
        
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        

    def transform_and_load_data(self):
        self.train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(p=0.2),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])


        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.valid_data = datasets.ImageFolder(self.valid_dir, transform=self.test_transforms)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True),
            'valid': torch.utils.data.DataLoader(self.valid_data, batch_size=64),
            'test': torch.utils.data.DataLoader(self.test_data, batch_size=64)
        }
        
        print('Finished transforming and loading data.')
        

    def get_pretrained_model(self):
        self.model = (getattr(models, self.arch))(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
            
        print('Finished getting pretrained model.')
        

    def build_classifier(self):
        classifier = nn.ModuleList()

        for idx, layer_units in enumerate(self.hidden_units):
            starting_int = self.model.classifier[0].in_features
            ending_int = 102

            if idx == 0:
                classifier.append(nn.Linear(starting_int, layer_units))
                classifier.append(nn.ReLU())
                classifier.append(nn.Dropout(p=0.2))

            if idx < len(args.hidden_units) - 1:
                classifier.append(nn.Linear(layer_units, self.hidden_units[idx+1]))
                classifier.append(nn.ReLU())
                classifier.append(nn.Dropout(p=0.2))

            if idx == len(self.hidden_units) - 1:
                classifier.append(nn.Linear(layer_units, ending_int))
                classifier.append(nn.LogSoftmax(dim=1))
                self.model.classifier = nn.Sequential(*classifier)
                
        print('Finished building classifier.')
        

    def train_and_validate(self):
        print('Starting training and validation.')
        self.device = torch.device('cuda' if self.gpu == True and torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print(f'Current epoch: {epoch + 1}')
            start_time = time.time()

            running_loss = 0

            for images, labels in self.dataloaders['train']:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            else:
                self.model.eval()

                valid_accuracy = 0
                valid_loss = 0

                with torch.no_grad():
                    for images, labels in self.dataloaders['valid']:
                        images, labels = images.to(self.device), labels.to(self.device)

                        output = self.model(images)
                        loss = self.criterion(output, labels)

                        valid_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                          f"Total Time: {(time.time() - start_time):.3f} seconds.."
                          f"Train loss: {running_loss/len(self.dataloaders['train']):.3f}.. "
                          f"Valid loss: {valid_loss/len(self.dataloaders['valid']):.3f}.. "
                          f"Valid accuracy: {valid_accuracy/len(self.dataloaders['valid']):.3f}")
                    self.model.train()
                    
        print('Finished training and validating.')

    def test(self):
        self.model.eval()

        test_accuracy = 0
        test_loss = 0

        with torch.no_grad():
            start_time = time.time()

            for images, labels in self.dataloaders['test']:
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, labels)

                test_loss += loss.item()

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss/len(self.dataloaders['test']):.3f}.. "
                  f"Test accuracy: {test_accuracy/len(self.dataloaders['test']):.3f}.."
                  f"Total Time: {(time.time() - start_time):.3f} seconds")
            self.model.train()
        
        print('Finished testing.')

    def save_checkpoint(self):
        state = self.model.state_dict()

        layers = [x.in_features for i, x in enumerate(self.model.classifier) if i % 3 == 0]

        # not using input_size, output_size, or hidden_layers in this case, but I like to store them
        torch.save({
            'input_size': layers[0],
            'output_size': layers[-1],
            'hidden_layers': layers[1:-1],
            'state_dict': state,
            'class_to_idx': self.train_data.class_to_idx
        }, f'{self.save_dir}/checkpoint.pth')
        print('Finished saving.')

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        
        self.get_pretrained_model()
        self.build_classifier()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.class_to_idx = checkpoint['class_to_idx']
        
        print('Finished loading checkpoint.')
        return self.model, self.class_to_idx
        
def main():
    model = Model(args.data_dir,
                  args.arch,
                  args.save_dir,
                  args.learning_rate,
                  args.hidden_units,
                  args.epochs,
                  args.gpu)
    
    model.transform_and_load_data()
    model.get_pretrained_model()
    model.build_classifier()
    model.train_and_validate()
    model.test()
    model.save_checkpoint()

    print('Model trained and saved!')
    
if __name__ == '__main__':
    main()