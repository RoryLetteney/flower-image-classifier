import argparse
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

from train import Model

parser = argparse.ArgumentParser(description="Train model of choice on data of choice.")
parser.add_argument('image_path', type=str, help='Path to image from root.')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint.pth')
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--gpu', type=bool, default=False)

args = parser.parse_args()

class Predict:
    def __init__(self, image_path, checkpoint_path, category_names='cat_to_name.json', top_k=5, gpu=False):
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.category_names = category_names
        self.top_k = top_k
        self.gpu = gpu
        
        self.model = Model(gpu=self.gpu)
        self.model, self.class_to_idx = self.model.load_checkpoint(checkpoint_path)
        
        
    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model#         returns an Numpy ary
        '''
    
        with Image.open(self.image_path) as img:
            img_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
            print('Finished processing image.')
            return torch.from_numpy(np.array([np.array(img_transforms(img))]))
        
    def load_category_names(self):
        with open(self.category_names) as cats:
            return json.load(cats)
    
    
    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        device = torch.device('cuda' if self.gpu == True and torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            img = self.process_image().type(torch.FloatTensor)

            images = img.to(device)

            output = self.model(images)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(self.top_k, dim=1)

            self.model.train()
            
            cats_to_name = self.load_category_names()
            idx_to_class = dict(map(reversed, self.class_to_idx.items()))
            
            print('Finished predicting.')
            
            top_p_list = top_p.cpu().numpy()[0]
            top_class_list = [cats_to_name[idx_to_class[x]] for x in top_class.cpu().numpy()[0]]
            
            for i in range(len(top_p_list)):
                print(f'{top_class_list[i]} has probability of {top_p_list[i]}')
        
        
def main():
    Predict(args.image_path,
            args.checkpoint_path,
            args.category_names,
            args.top_k,
            args.gpu).predict()
    
if __name__ == '__main__':
    main()
