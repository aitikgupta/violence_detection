# Violence Detection using VGG16 network
This is a Computer Vision project which aims to classify images containing violence.<br>Major Tech Stack:<br>->TensorFlow 2.1.0<br>->OpenCV


# How to use?
Note: To maintain the ennvironments, I highly recommend using [conda](https://anaconda.org/).
```
git clone https://github.com/aitikgupta/violence_detection.git
cd violence_detection
conda env create -f environment.yml
conda activate {environment name, for eg. conda activate cv}
jupyter notebook Training_Model.ipynb {or you can use my trained model, link is below}
jupyter notebook Violence_Detection.ipynb
```
## Model link:

[https://drive.google.com/file/d/1ib6zg_8kWmRQhkVFRszi3Y6r1fB5jR1a/view?usp=sharing](https://drive.google.com/file/d/1ib6zg_8kWmRQhkVFRszi3Y6r1fB5jR1a/view?usp=sharing)

Note: If you choose to download my trained model, place the model.h5 file in the root directory.

## About the model:

The model is built using TensorFlow 2.1.0 with 1.5 hours training on GeForce GTX-1650 GPU. Last 4 layers of the VGG16 pretrained network were fine tuned, along with fully-connected layers.


#### Application of VGG16 network:

Given image → find object name in the image
It can detect any one of 1000 images
It takes input image of size 224 * 224 * 3 (RGB image)
Built using:

Convolutions layers (used only 3*3 size )
Max pooling layers (used only 2*2 size)
Fully connected layers at end
Total 16 layers
#### Model size:
528MB

#### Pre trained model(Tensorflow):
[VGG16-weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)

#### Built by:
Visual Geometry Group [VGG Homepage](http://www.robots.ox.ac.uk/~vgg/)

#### Description of layers:

![Couldn't find image!](https://github.com/aitikgupta/violence_detection/blob/master/Screenshots/Network/network1.png)

Convolution using 64 filters<br>
Convolution using 64 filters + Max pooling<br>
Convolution using 128 filters<br>
Convolution using 128 filters + Max pooling<br>
Convolution using 256 filters<br>
Convolution using 256 filters<br>
Convolution using 256 filters + Max pooling<br>
Convolution using 512 filters<br>
Convolution using 512 filters<br>
Convolution using 512 filters + Max pooling<br>
Convolution using 512 filters<br>
Convolution using 512 filters<br>
Convolution using 512 filters + Max pooling<br>
Fully connected with 4096 nodes<br>
Fully connected with 4096 nodes<br>
Output layer with Softmax activation with 1000 nodes<br>
#### Full view at image level:

![Couldn't find image!](https://github.com/aitikgupta/violence_detection/blob/master/Screenshots/Network/network2.png)

