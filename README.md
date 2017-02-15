##Usage
* Download ImageNet dataset [here](http://www.image-net.org/challenges/LSVRC/2015/).
You will have to register in order to obtain permission.
* Point `DATA_DIR` in `prepare_imagenet.py` to the data directory and run data preparation.
This will generate `DATA_DIR/numpy/val_data.hdf5` file.
```
$ python prepare_imagenet.py
```
* Download model weights [here](http://elbereth.zemris.fer.hr/kivan/resnet/).
These weights were converted from original Caffe models using
tensorpack converter from [here](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet).
* Configure paths to data and model weights in `resnet_imagenet.py`
```
MODEL_DEPTH=50
MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
DATA_PATH = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'
```
* Run evaluation:
```
& python resnet_imagenet.py
```
