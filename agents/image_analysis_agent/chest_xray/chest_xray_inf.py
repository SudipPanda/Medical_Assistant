import logging
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ChestXray:
    def __init__(self , model_path , device=None):
        self.logger = logging.getLogger(__name__)
        self.class_names = ['covid19', 'normal']
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.logger.info(f"Using device: {self.device}")
        
        #Load Model Here
        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]
        self.transform = transforms(
            transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_nums, std=self.std_nums)
        ])
        )

        #load the model here 
        self.model = self._get_model(weights = None)
        self._load_weight(model_path=model_path)
        self.model.to(self.device)
        self.model.eval()
        
    
    def _get_model(self , wights = None):
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(self.class_names))
        return model
    
    def _load_weight(self , model_path):
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"The load of model weight is successfull here")
        
        except Exception as e:
            self.logger.error(f"the error here is {e}")
            raise e
    
    def predict(self , image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                out = self.model(image_tensor)
                _, preds = torch.max(out, 1)
                idx = preds.cpu().numpy()[0]
                pred_class = self.class_names[idx]
            
            return pred_class
        
        except Exception as e:
            self.logger.error(f"an error has occure here {e}")
            raise e

