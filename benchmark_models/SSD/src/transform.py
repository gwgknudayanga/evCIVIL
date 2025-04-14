"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import random
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
#import torch.nn.functional as F
import torchvision.transforms.functional as F
from src.utils import Encoder
import numpy as np


class SSDCropping(object):
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img, img_size, bboxes, labels):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            original_boxes = bboxes

            for _ in range(1):
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:
                    continue

                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                #print("kkkkkkkkkkkkkkkkk ",bboxes," ",bboxes.shape)
                #print("llllllllllllllll ",torch.tensor([[left, top, right, bottom]]).shape)
                
                ious = box_iou(bboxes, torch.tensor([[left, top, right, bottom]]))

                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
        
                img = img[top_idx:bottom_idx,left_idx:right_idx,:] #HCW format of the array
                
                #img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                
                return img, (htot, wtot), bboxes, labels


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        
        if random.random() < self.prob: 
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            img = np.flip(img,axis=1).copy() #he do the lr flipping in the dimension 1
            #return img.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, bboxes

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self,img,bboxes):
        if random.random() < self.prob:
            bboxes[:, 1], bboxes[:,3] = 1.0 - bboxes[:, 3], 1.0 - bboxes[:, 1]
            img = np.flip(img,axis=0).copy() #for flipud the it does the up down in 0 dimesnion
            #return img.transpose(Image.FLIP_TOP_BOTTOM), bboxes
        return img, bboxes

class SSDTransformer(object):
    def __init__(self, dboxes, size=(300, 300), val=False,input_image_type = 0):
        self.size = size
        self.val = val
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.crop = SSDCropping()
        self.input_image_type = input_image_type

        self.hflip = RandomHorizontalFlip()
        self.vflip = RandomVerticalFlip()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        """self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            self.normalize
        ])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize
        ])"""

        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        
        
    def resizeToTargetSize(self,img):
        img = F.resize(img.unsqueeze(0), self.size)
        #img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear') #This requires since cropping nin
        img = img.squeeze(0)
        
        return img
    
    def preprocess_and_resize_input_img_tensor(self,img,input_image_type):

        if input_image_type == 0:
            img = img.permute(2,0,1)
            img = self.resizeToTargetSize(img)
            img = img/255.
        elif input_image_type == 1:
            img = img.permute(2,0,1) 
            img = self.resizeToTargetSize(img)
            img = img/255.
        elif input_image_type == 2: #check whether the histogram is in CHW or HWC format
            img = img.permute(2,0,1)
            img = self.resizeToTargetSize(img)
            #This is like normalize before resize
        return img

        
    def __call__(self, img, img_size, bboxes=None, labels=None, max_num=200):
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bboxes.size(0), :] = bboxes
            label_out[:labels.size(0)] = labels
            img = torch.tensor(img,dtype = torch.float32)
            img = self.preprocess_and_resize_input_img_tensor(img,self.input_image_type)
            return img, img_size, bbox_out, label_out

        
        img, img_size, bboxes, labels = self.crop(img, img_size, bboxes, labels)
        img, bboxes = self.hflip(img, bboxes)
        img, bboxes = self.vflip(img, bboxes)
        
        #img = self.preprocess_img(img)
        img = torch.tensor(img,dtype = torch.float32)
        img = self.preprocess_and_resize_input_img_tensor(img,self.input_image_type)
        img = img.contiguous()
        bboxes, labels = self.encoder.encode(bboxes, labels)
        return img, img_size, bboxes, labels
