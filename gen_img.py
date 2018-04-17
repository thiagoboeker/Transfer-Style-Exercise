
#Imports
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, Model
from content import Content
from style import Style
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from scipy.optimize import fmin_l_bfgs_b
import keras.backend as K
import utils_c
import matplotlib.pyplot as plt
import matplotlib.image as pltimage

if __name__ == "__main__":

    #Paths
    img_content_path = 'thiago.jpg'
    img_style_path = 'star_night.jpg'

    #instantiate the classes
    content = Content(img_content_path, 11)
    style = Style(img_style_path, content.batch_shape)
    content.image = utils_c.Load_and_resize(img_content_path, (style.image.shape[1], style.image.shape[2]))
    content.batch_shape = content.image.shape
    
    #VGG
    VGGpool = utils_c.VggAvgPOOL(content.batch_shape)
    
    #Merged Model
    contentModel = Model(VGGpool.input, VGGpool.layers[13].get_output_at(1))
    #Target
    content_target = K.variable(contentModel.predict(content.image))

    #The layer.get_output_at(1) refer to the index of the VGGpool diferent nets
    #Get the output of the first conv of each block for the style
    s_outputs = [layer.get_output_at(1) for layer in VGGpool.layers if layer.name.endswith('conv1')]

    #Style Model
    styleModel = Model(VGGpool.input, s_outputs)
    
    
    stl_model_ouputs = [K.variable(y) for y in styleModel.predict(style.image)]

    #Weights of each block output in the result
    style_weights = [0.1,0.2,0.3,0.4,0.5]

    #Content Loss
    loss = K.mean(K.square(contentModel.output - content_target))
    
    #Sum of the content Loss and the Style Loss
    #The style loss is the MSE of the gram-matrix of both outputs 
    for w, stl, conv in zip(style_weights, s_outputs, stl_model_ouputs):
        loss += w*style.style_Loss(conv[0], stl[0])
    
    #Get the gradients
    grads = K.gradients(loss, VGGpool.input)
    
    #A Keras function to pass to the multi-diff scipy function
    get_loss_and_grads_content = K.function(inputs = [VGGpool.input], outputs = [loss] + grads)

    def get_loss_grads_wraper(x):
        l, g = get_loss_and_grads_content([x.reshape(*content.batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    #The image that will be generated
    x = np.random.randn(np.prod(content.batch_shape))
    losses = []
    for i in range(10):
        #Call to the scipy.optimize multidiff function
        x, l, _ = fmin_l_bfgs_b(func = get_loss_grads_wraper, x0 = x, maxfun = 20)
        #Clip x
        x = np.clip(x, -127, 127)
        losses.append(l)
        print("Epoch:{} loss:{}".format(i, l))

    #Print the image
    final_img = x.reshape(*content.batch_shape)
    final_img = utils_c.unprocess_input(final_img)
    pltimage.imsave("thiago_star_night2.jpg",utils_c.scale_img(final_img[0]))
    plt.imshow(utils_c.scale_img(final_img[0]))
    plt.show()