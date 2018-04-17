from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import MaxPooling2D, AveragePooling2D

#Standard load method
def Load_Image(img_path):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

#Load image resizing if necessary, just to make more clear
def Load_and_resize(img_path, shape = None):
    img = image.load_img(img_path, target_size = shape)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

#Standard VGG16 average pooling instead of max pooling
def VggAvgPOOL(batch_shape):
    vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = batch_shape[1:])
    avg = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            avg.add(AveragePooling2D())
        else:
            avg.add(layer)
    return avg

#Reverse the VGG16 preprocess_input
def unprocess_input(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.979            
    img[..., 2] += 128.88
    img = img[..., ::-1]
    return img

#Scale the generated image between 0-1 for ploting    
def scale_img(img):
    img -= img.min()
    img /= img.max()
    return img

