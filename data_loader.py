from __future__ import print_function, absolute_import

import os

from PIL import Image
from torch.utils.data import Dataset

from path import DATA_PATH, DataID


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
            # print("sucess")
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageData(Dataset):
    def __init__(self, path_list,label_list, transform):
        self.dataset = path_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item]
        label = self.label[item]
        img = read_image(os.path.join(DATA_PATH,DataID,img))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

