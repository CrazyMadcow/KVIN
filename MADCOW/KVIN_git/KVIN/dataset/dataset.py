import numpy as np

import torch
import torch.utils.data as data


class GridworldData(data.Dataset):
    def __init__(self,
                 file,
                 imsize,
                 train=True,
                 transform=None,
                 target_transform=None):
        assert file.endswith('.npz')  # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.images, self.S1, self.S2, self.Gamma, self.labels =  \
                                self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        gamma = self.Gamma[index]
        label = self.labels[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:  # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        # Set proper datatypes
          # (S1, S2) are float
        label.astype(np.float32)
          # labels are long
        return img, float(s1), float(s2), float(gamma), label

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                Gamma = f['arr_3']
                labels = f['arr_4']
            else:
                images = f['arr_5']
                S1 = f['arr_6']
                S2 = f['arr_7']
                Gamma = f['arr_8']
                labels = f['arr_9']

        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(np.float32)  # (S1, S2) location are float
        S2 = S2.astype(np.float32)
        Gamma = Gamma.astype(np.float32)
        labels = labels.astype(np.float32)  # labels are float
        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, Gamma, labels
