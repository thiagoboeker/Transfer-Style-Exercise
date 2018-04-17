
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from pathlib import Path
from keras.layers.convolutional import Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.preprocessing import image
from scipy.optimize import fmin_l_bfgs_b
import keras.backend as K
import utils_c

#Class that hold the content
class Content:

    def __init__(self, img_path, num_convs = None):
        
        self.img_path = img_path
        self.image = utils_c.Load_Image(self.img_path)
        self.batch_shape = self.image.shape
        self.num_convs = num_convs
    
    #A cut off version of VGG16
    def VggMinLayers(self):
        avg = utils_c.VggAvgPOOL(self.batch_shape)
        n = 0
        model = Sequential()
        for layer in avg.layers:
            if layer.__class__ == Conv2D:
                n+=1
            model.add(layer)
            if n >= self.num_convs:
                break
        return model

# Code to test the content, the output image should be a image no aparent color("style"), but sharp enough to
# to ressemble the input image
if __name__ == "__main__":
    img_path = 'cat.jpg'
    content = Content(img_path, num_convs=11)
    
    ContentVgg = content.VggMinLayers()
    ContentVgg.summary()
    target = K.variable(ContentVgg.predict(content.image))
    loss = K.mean(tf.square(target - ContentVgg.output))
    grads = K.gradients(loss, ContentVgg.input)

    get_grads_loss = K.function(inputs = [ContentVgg.input], outputs = [loss] + grads)
    
    def get_grads_loss_wraper(x):
        l, g = get_grads_loss([x.reshape(*content.batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    losses = []
    
    x = np.random.randn(np.prod(content.batch_shape))
    print("Starting...")
    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(func = get_grads_loss_wraper, x0 = x, maxfun=20)
        x = np.clip(x, -127, 127)
        losses.append(l)
        print("Epoch:{} loss:{}".format(i, l))
    
    plt.plot(losses)
    plt.show()
    shape_x = content.batch_shape[1:]
    x = x.reshape(*shape_x)
    final_img = utils_c.unprocess_input(x)
    plt.imshow(utils_c.scale_img(final_img))
    plt.show()
