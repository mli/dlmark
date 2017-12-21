from __future__ import absolute_import
import numpy as np
# from . import data #import DownloadMultiPartDataset
from .data import DownloadMultiPartDataset


def preprocess_imagenet_val(image_dir, label_fname, output_dir, image_shape):
    """Save raw Imagenet validation images into numpy format
    """
    import mxnet as mx



class ILSVRC12Val(object):
    def __init__(self, batch_size, repo_dir='',
                 root='~/.dlmark/datasets/ilsvrc12_val'):
        self.dataset = DownloadMultiPartDataset(repo_dir, root)
        self.batch_size = batch_size
        self.num_examples = 50000
        self.num_examples_per_part = 1000
        self.curr_part = -1
        self.X = None
        self.y = None
        assert batch_size < self.num_examples_per_part

    def _load_part(self, part):
        fname = self.dataset[part]
        data = np.load(fname)
        self.X = data['X']
        self.y = data['y']
        self.curr_part = part

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        part = idx*self.batch_size//self.num_examples_per_part
        if part != self.curr_part:
            self._load_part(part)
        offset = idx * self.batch_size - part * self.num_examples_per_part
        X, y = (self.X[offset:offset+self.batch_size],
                self.y[offset:offset+self.batch_size])

        if X.shape[0] < self.batch_size:
            self._load_part(part+1)
            n = self.batch_size - X.shape[0]
            X = np.concatenate((X, self.X[:n]), axis=0)
            y = np.concatenate((y, self.y[:n]), axis=0)
        return (X, y)

    def __len__(self):
        return self.num_examples // self.batch_size

if __name__ == '__main__':
    data = ILSVRC12Val(128, 'http://xx/', root='/home/ubuntu/imagenet_val/')
    print(data[0][0].shape)
    print('xxx')
