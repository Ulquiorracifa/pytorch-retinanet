'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop
import xml.etree.ElementTree as ET
import numpy as np


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, xml_file, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []
        # self.image_index = []
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))

        self.encoder = DataEncoder()

        with open(list_file, 'r') as f:
            self.fnames = [x.strip() for x in f.readlines()]
            self.num_samples = len(self.fnames)
        # with open(list_file) as f:
        #     lines = f.readlines()
        #     self.num_samples = len(lines)

        # for line in lines:
        #     splited = line.strip().split()
        #     self.fnames.append(splited[0])
        #     num_boxes = (len(splited) - 1) // 5
        #     box = []
        #     label = []
        #     for i in range(num_boxes):
        #         xmin = splited[1+5*i]
        #         ymin = splited[2+5*i]
        #         xmax = splited[3+5*i]
        #         ymax = splited[4+5*i]
        #         c = splited[5+5*i]
        #         box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
        #         label.append(int(c))
        #     self.boxes.append(torch.Tensor(box))
        #     self.labels.append(torch.LongTensor(label))
        for index in self.fnames:
            # label = np.zeros((self.cell_size, self.cell_size, 25))
            filename = os.path.join(xml_file, index + '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')
            # w_ratio,h_ratio = 1,1

            for obj in objs:
                box = []
                label = []
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                # x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
                # y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
                # x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
                # y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
                # boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                # x_ind = int(boxes[0] * self.cell_size / self.image_size)
                # y_ind = int(boxes[1] * self.cell_size / self.image_size)
                box.append([x1, y1, x2, y2])
                label.append(int(cls_ind))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
            # if label[y_ind, x_ind, 0] == 1:
            #     continue
            # label[y_ind, x_ind, 0] = 1
            # label[y_ind, x_ind, 1:5] = boxes
            # label[y_ind, x_ind, 5 + cls_ind] = 1




    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname+".jpg"))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='/mnt/hgfs/D/download/PASCAL_VOC/voc_all_images',
                          list_file='./data/voc12_train.txt', train=True, transform=transform, input_size=400)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

# test()
