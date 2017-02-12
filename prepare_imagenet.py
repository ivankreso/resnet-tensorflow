import os

from tqdm import trange
import numpy as np
import h5py
import cv2


DATA_DIR = '/home/kivan/datasets/imagenet/ILSVRC2015/'

save_path = os.path.join(DATA_DIR, 'numpy', 'val_data.hdf5')
imglist_fp = open('val.txt', 'r')
img_list = []
img_class = []
for line in imglist_fp:
  lst = line.strip().split(' ')
  img_list += [lst[0]]
  img_class += [int(lst[1])]
N = len(img_list)
data_y = np.zeros((N), dtype=np.int32)
for i, c in enumerate(img_class):
  data_y[i] = c

img_dir = os.path.join(DATA_DIR, 'Data/CLS-LOC/val')
labels_dir = os.path.join(DATA_DIR, 'Annotations/CLS-LOC/val/')

resize_size = 256
crop_size = 224

N = len(img_list)
data_x = np.zeros((N, crop_size, crop_size, 3), dtype=np.uint8)

for i in trange(N):
  img_name = img_list[i]
  rgb_path = os.path.join(img_dir, img_name)
  rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
  height = rgb.shape[0]
  width = rgb.shape[1]
  if height > width:
    new_w = resize_size
    new_h = int(round(resize_size * (height/width)))
  else:
    new_h = resize_size
    new_w = int(round(resize_size * (width/height)))
  height, width = new_h, new_w

  rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
  assert resize_size > crop_size
  ys = int((height - crop_size) * 0.5)
  xs = int((width - crop_size) * 0.5)
  rgb = np.ascontiguousarray(rgb[ys:ys+crop_size,xs:xs+crop_size,:])
  data_x[i] = rgb

h5f = h5py.File(save_path, 'w')
h5f.create_dataset('data_x', data=data_x)
h5f.create_dataset('data_y', data=data_y)
dt = h5py.special_dtype(vlen=str)
img_list = np.array(img_list, dtype=object)
h5f.create_dataset('img_names', data=img_list, dtype=dt)
h5f.close()
