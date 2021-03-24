"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import math
import tensorflow as tf

try:
  xrange
except:
  xrange = range
  
FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True) # read as YCrCb format
  label_ = modcrop(image, scale) # crop 使得整除于3

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.
  
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  image_formats = ["*.jpg", "*.bmp", "*.png"]
  data = []
  if FLAGS.is_train:
    data_dir = os.path.join(os.getcwd(), dataset)
    for img_format in image_formats:
        data.extend(glob.glob(os.path.join(data_dir, img_format))) 
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), FLAGS.dev_dir)
    for img_format in image_formats:
        data.extend(glob.glob(os.path.join(data_dir, img_format)))
  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(sess, config, test_image_path=None):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  if TEST return
  nx : patch number over x direction,
  ny : patch number over y direction,
  input_ : input image with bibicu 
  label_: ground_turth
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train")
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []

  if config.is_train:
    for i in xrange(len(data)):
      input_, label_ = preprocess(data[i], config.scale)
      input_sequence, label_sequence, nx, ny = imslice(input_, label_, config)
      sub_input_sequence.extend(input_sequence)
      sub_label_sequence.extend(label_sequence)
  else:
    input_, label_ = preprocess(test_image_path, config.scale) 
    sub_input_sequence, sub_label_sequence, nx, ny = imslice(input_, label_, config)
  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
    return nx, ny, input_, label_

def imslice(input_, label_, config):
    """
    slice input and  to indicated size
    return subinput sequences and label sequences, and nx, ny 
    """
    sub_input_sequence = []
    sub_label_sequence = []

    padding = abs(config.image_size - config.label_size) / 2 # 6

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape
    nx = ny = 0

    for x in range(0, h-config.image_size+1, config.stride):
        nx += 1; ny = 0
        for y in range(0, w-config.image_size+1, config.stride):
            ny += 1
            sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
            sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]

            # Make channel value
            sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)
        
    return sub_input_sequence, sub_label_sequence, nx, ny
def imsave(image, path):
  return scipy.misc.imsave(path, image)

def toimage(image):
    return scipy.misc.toimage(image, channel_axis=2)

def PSNR(pred, ground_truth):
    """
    pred, ground_truth both are (?, fsub, fsub, 1) array
    return PSNR of pred and center of gt
    """
    pred_shape = pred.shape
    gt_shape = ground_truth.shape
    if len(pred_shape) < 3 or len(gt_shape) < 3:
        return -1 # error
    padding = abs(pred_shape[1] - gt_shape[1]) // 2 # 6
    pred_ = np.reshape(pred, (pred_shape[0], pred_shape[1], pred_shape[2]))
    gt = np.reshape(ground_truth, (gt_shape[0], gt_shape[1], gt_shape[2]))

    if gt.shape[1] > pred_.shape[1]:
        gt = gt[:, padding:gt_shape[1] - padding, padding:gt_shape[2] - padding]
    elif pred_.shape[1] > gt.shape[1]:
        pred_ = pred_[:, padding:pred_shape[1] - padding, padding:pred_shape[2] - padding]

    imdiff = gt - pred_
    rmse = math.sqrt(np.mean(imdiff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)






def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img
