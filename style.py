
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from pathlib import Path
from keras.models import Sequential, Model
from scipy.optimize import fmin_l_bfgs_b
import keras.backend as K
from datetime import datetime
import utils_c

#Class for the style
class Style:

    def __init__(self, img_path, content_shape):
        
        self.img_path = img_path
        self.content_shape = content_shape
        self.image = utils_c.Load_Image(self.img_path)
        self.batch_shape = self.image.shape

    #Seek the correlation between an item with itself, X.XT/N
    def gram_matrix(self,x):
        y = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
        g = K.dot(y, K.transpose(y))/x.get_shape().num_elements()
        return g
    
    #The MSE of the target vs the output with their gram matrixs
    def style_Loss(self, x, y):
        return K.mean(K.square(self.gram_matrix(y) - self.gram_matrix(x)))
    
    def train(self, func, epochs, shape):
        losses = []
        x = np.random.randn(np.prod(shape))
        for i in range(epochs):
            x, l, _ = fmin_l_bfgs_b(func = func, x0 = x, maxfun = 20)
            x = np.clip(x, -127, 127)
            losses.append(l)
            print("Epoch:{}  loss:{}".format(i, l))
        n_img = x.reshape(*shape)
        final_img = utils_c.unprocess_input(n_img)
        return final_img

# Code to check the Style class the final_img should output a image with no sharpness but showing the style of the
# given image with color pattern e etc
if __name__ == "__main__":

    img_path = 'star_night.jpg'

    style = Style(img_path, None)
    print(style.image.shape)
    styleVGG = utils_c.VggAvgPOOL(style.batch_shape)
    styleVGG.summary()
    conv_outputs = [layer.get_output_at(1) for layer in styleVGG.layers if layer.name.endswith('conv1')]
    merged_model = Model(styleVGG.input, conv_outputs)
    style_outputs = [K.variable(y) for y in merged_model.predict(style.image)]
    loss = 0
    for stl, conv in zip(style_outputs, conv_outputs):
        print(conv[0], stl[0])
        loss += style.style_Loss(conv[0], stl[0])
    
    grads_style = K.gradients(loss, merged_model.input)

    get_grads_loss = K.function(inputs = [merged_model.input], outputs = [loss] + grads_style)

    def get_grads_loss_style_wraper(x):
        l, g = get_grads_loss([x.reshape(*style.batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    print("Starting......")
    final_img = style.train(get_grads_loss_style_wraper, 10, style.batch_shape)
    plt.imshow(utils_c.scale_img(final_img[0]))
    plt.show()







        





