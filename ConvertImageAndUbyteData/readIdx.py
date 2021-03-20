import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
from numpy import genfromtxt
from glob import glob
import pandas as pd

#https://github.com/nyanp/mnisten
def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s" % glob_files)
    dataset = []
    for file_count, file_name in enumerate(sorted(glob(glob_files), key=len)):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA')  # tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed" % file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels, names=["class"])
        return np.array(dataset), np.array(df["class"])
    else:
        return np.array(dataset)


#Data1, y1 = dir_to_dataset("train/*.png", "")


i=0

import os
import gzip
import shutil
import struct
from os import path as osp
#with open('download/gzip/emnist-balanced-test-images-idx3-ubyte','rb') as f:
with gzip.open(os.path.join("Handwriting_Recognition-master/mnist", 'train-labels-idx1-ubyte.gz'), 'rb') as zipped:
    with open(osp.splitext(os.path.join("Handwriting_Recognition-master/mnist", 'train-labels-idx1-ubyte.gz'))[0], mode='wb') as unzipped:
        shutil.copyfileobj(zipped, unzipped)

with open('Handwriting_Recognition-master/mnist/train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    print (magic)
    print (size)
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))
    if i==0:
        plt.imshow(data[0, :, :], cmap='gray')
        plt.show()
    i=i+1

