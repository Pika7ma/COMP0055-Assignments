import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys, pylab, operator, csv
import os
import scipy.io as sio
import sys

# replace the following path with your caffe path
sys.path.insert(0, '/home/likewise-open/SENSETIME/mapingchuan/Projects/caffe/')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('solver.prototxt')

dim = 1024

solver.net.blobs['input_feat'].reshape(1, 1, dim, dim)
solver.net.blobs['input_feat'].data[0][0] = np.random.randn(dim, dim)

solver.net.forward()
print "mpc max pooling forward done"
