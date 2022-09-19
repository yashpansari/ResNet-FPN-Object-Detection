import typing as t
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data
import cv2
from torchsummary import summary
import math


class StarModelResNet(nn.Module):
    """
    ResNet Bakcbone for FPN Architecture - inspired by Jonathan Hui, Medium, 2018 - Understanding FPNs for Object Detection
    (https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
    """
    def __init__(self, channels_in, channels_out, padding=1, stride=2):
        super(StarModelResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels_out)
        self.act1 = nn.ReLU()  # Activation function
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels_out)
        self.dnsmpl = None  # Down-sampling

        if stride != 1:
            self.dnsmpl = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(channels_out))

    def forward(self, x):
        skip_connection = x
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        skip_connection = self.dnsmpl(skip_connection)
        x = x + skip_connection
        x = self.act1(x)
        return x


class StarModelFPN(nn.Module):
    """
    Feature Pyramid Networks for Object Detection - Lin et. al. (2016), Facebook AI Research
    (https://arxiv.org/abs/1612.03144 https://github.com/Aditib2409/)
    """
    def __init__(self, use_last = False):
        self.save_fp = "./model.pickle"
        super(StarModelFPN, self).__init__()

        output_filters = [16, 32, 64, 128, 256]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, output_filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_filters[0]),
            nn.MaxPool2d(2)
        )

        # Pyramid channels
        pyramid_channels = 256

        # Bottom-up channels
        self.conv_2 = StarModelResNet(output_filters[0], output_filters[1], stride=2, padding=1)
        self.conv_3 = StarModelResNet(output_filters[1], output_filters[2], stride=2, padding=1)
        self.conv_4 = StarModelResNet(output_filters[2], output_filters[3], stride=2, padding=1)
        self.conv_5 = StarModelResNet(output_filters[3], output_filters[4], stride=2, padding=1)

        self.pyramid_bottleneck = nn.Conv2d(output_filters[4], pyramid_channels, 1)

        # Reduced and smoothing bottom-up channels
        self.conv_2_reduced = nn.Conv2d(output_filters[1], pyramid_channels, 1)
        self.conv_3_reduced = nn.Conv2d(output_filters[2], pyramid_channels, 1)
        self.conv_4_reduced = nn.Conv2d(output_filters[3], pyramid_channels, 1)

        self.conv_2_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)
        self.conv_3_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)
        self.conv_4_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Boolean classifier for "has_star"
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(pyramid_channels, 1))

        # Regressor to output the x, y, theta, w, h
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(pyramid_channels, 5))

        self.sigmoid = nn.Sigmoid()

        if use_last:
          self.load_params()

    def forward(self, x, train = False):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.conv_3(out2)
        out4 = self.conv_4(out3)
        out5 = self.conv_5(out4)

        pyramid_out5 = self.pyramid_bottleneck(out5)
        pyramid_out4 = self.conv_4_reduced(out4) + F.interpolate(pyramid_out5, size=out4.shape[-2:], mode="nearest")
        pyramid_out3 = self.conv_3_reduced(out3) + F.interpolate(pyramid_out4, size=out3.shape[-2:], mode="nearest")
        pyramid_out2 = self.conv_2_reduced(out2) + F.interpolate(pyramid_out3, size=out2.shape[-2:], mode="nearest")

        classifier_features = self.average_pooling(pyramid_out5)
        regressor_features = self.average_pooling(pyramid_out5)

        classification_output = self.classifier(classifier_features)
        classification_output = self.sigmoid(classification_output)

        regression_output = self.regressor(regressor_features)
        regression_output = self.sigmoid(regression_output)

        prediction = classification_output.view(x.shape[0], 1)

        bounding_box = regression_output.view(x.shape[0], 5)
        
        if not train:
          bounding_box[:,2] *= 3.1415926/100
          bounding_box*=200
          for i in range(len(prediction)):
            if prediction[i,0] < 0.1:
              for j in range(5):
                bounding_box[i,j] = np.NaN
          return bounding_box
        return torch.cat((prediction, bounding_box), dim=1)
        
    def load_params(self):
      self.load_state_dict(torch.load(self.save_fp, map_location = torch.device(DEVICE)))

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data()
        return image[None], label


def train(model: StarModelFPN, dl: StarDataset, num_epochs: int, loss_fn) -> StarModelFPN:

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            optimizer.zero_grad()

            preds = model(image, train = True)
            loss = loss_fn(label, preds)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        torch.save(model.state_dict(), model.save_fp)
        print(np.mean(losses))
        
    return model
    
    
"""
Loss Function devised by me with no inspiration or citations.
"""

def box(t):
    """
    Args:
    t : A tensor of shape (N,5) indicating the [x, y, yaw, width, height], activated using nn.sigmoid
      
    Returns:
    Constructs a bounding box and returns the tensor of shape (N,8) indicating [ax, ay, bx, by, cx, cy, dx, dy], the x and y coordinates of the corners of the bounding box
    """
      
    w = 200*t[:,-2]
    h = 200*t[:,-1]
      
    #normalises yaw to the interval [0, pi] considering the rotational symmetry of rectangles
    theta = 2*3.141593*torch.where(t[:,2] >= 0.5, t[:,2] - 0.5, t[:,2])
      
    #Calculates vectors from the center, corresponding to the two diagonals of the rectangle
    dx1 = 0.5*(w*torch.cos(theta) + h*torch.sin(theta))
    dx2 = 0.5*(-w*torch.cos(theta) + h*torch.sin(theta))
    dy1 = 0.5*(h*torch.cos(theta) - w*torch.sin(theta))
    dy2 = 0.5*(h*torch.cos(theta) + w*torch.sin(theta))
      
    return torch.column_stack((200*t[:,0]+dx1,200*t[:,1]+dy1,200*t[:,0]+dx2,200*t[:,1]+dy2,200*t[:,0]-dx1,200*t[:,1]-dy1,200*t[:,0]-dx2,200*t[:,1]-dy2))

def box2(t):
    """
    Args:
    t : A tensor of shape (N,5) indicating the [x, y, yaw, width, height], in the bounds actually intended.
      
    Returns:
    Constructs a bounding box and returns the tensor of shape (N,8) indicating [ax, ay, bx, by, cx, cy, dx, dy], the x and y coordinates of   the corners of the bounding box
    """
      
    w = t[:,-2]
    h = t[:,-1]
      
    #normalises yaw to the interval [0, pi] considering the rotational symmetry of rectangles
    theta = torch.where(t[:,2] >= 3.141593, t[:,2] - 3.141593, t[:,2])
      
    #Calculates vectors from the center, corresponding to the two diagonals of the rectangle
    dx1 = 0.5*(w*torch.cos(theta) + h*torch.sin(theta))
    dx2 = 0.5*(-w*torch.cos(theta) + h*torch.sin(theta))
    dy1 = 0.5*(h*torch.cos(theta) - w*torch.sin(theta))
    dy2 = 0.5*(h*torch.cos(theta) + w*torch.sin(theta))
      
    return torch.column_stack((t[:,0]+dx1,t[:,1]+dy1,t[:,0]+dx2,t[:,1]+dy2,t[:,0]-dx1,t[:,1]-dy1,t[:,0]-dx2,t[:,1]-dy2))

def Loss_function(target, prediction):

    assert target.shape[-1] == 5
    assert prediction.shape[-1] == 6

    
    target = torch.cat((torch.where(torch.isnan(target[:,0]),0,1).unsqueeze(-1),target),1).to(DEVICE)
    
    # MAE loss for co-ordinates of the bounding boxes
    loss_bounding_box = torch.sum(torch.abs(box(prediction[:, 1:]) - box2(target[:, 1:])), dim = 1)
    
    # Ignore regression loss where there is no star
    no_star_index = torch.nonzero(target[:, 0] == 0, as_tuple=True)
    loss_bounding_box[no_star_index] = 0
    
    # BCE loss for classification
    loss_classification = torch.nn.BCELoss(reduction='none')(prediction[:, 0], target[:, 0])
    
    # Scale up classification to increase gradients on classifier
    loss_total = torch.sum(loss_bounding_box + loss_classification)
    
    return loss_total

def Loss_phase_3(target, prediction):

    assert target.shape[-1] == 5
    assert prediction.shape[-1] == 6

    target = torch.cat((torch.where(torch.isnan(target[:,0]),0,1).unsqueeze(-1),target),1).to(DEVICE)

    # MAE loss for co-ordinates of the bounding boxes
    loss_bounding_box = torch.sum(torch.abs(box(prediction[:, 1:]) - box2(target[:, 1:])), dim = 1)
    
    # Ignore regression loss where there is no star
    no_star_index = torch.nonzero(target[:, 0] == 0, as_tuple=True)
    loss_bounding_box[no_star_index] = 0

    # BCE loss for classification
    loss_classification = torch.nn.BCELoss(reduction='none')(prediction[:, 0], target[:, 0])
    
    # Scale up classification to increase gradients on classifier
    loss_total = torch.sum(loss_bounding_box + 40*loss_classification)
    
    return loss_total
    
def main():

    model = StarModelFPN().to(DEVICE)
    inpt = torch.rand((2, 1, 200, 200))
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=16),
        num_epochs=150,
        loss_fn = Loss_phase_3
    )
    with open('./model_summary.txt', 'w') as f:
        summary_report = summary(model, input_size = (1, 200, 200))
        f.write(str(summary_report))
        
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
