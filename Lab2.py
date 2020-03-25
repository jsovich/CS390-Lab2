import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
import imageio
from scipy.optimize import \
    fmin_l_bfgs_b  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "Content/NYC.jpg"
STYLE_IMG_PATH = "Style/Wave.jpeg"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.003  # Alpha weight.
STYLE_WEIGHT = 3.0  # Beta weight.
TOTAL_WEIGHT = .75

TRANSFER_ROUNDS = 10

# =============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4.0 * (3 ^ 2) * ((STYLE_IMG_H * STYLE_IMG_W) ^ 2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    a = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# =========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return (cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W)


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = img.reshape(ih, iw, 3)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")
    for layer_name in styleLayerNames:
        layer_features = outputDict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = styleLoss(style_reference_features, combination_features)
        loss = loss + (STYLE_WEIGHT / len(styleLayerNames)) * sl
    loss += TOTAL_WEIGHT * totalLoss(genTensor)
    gradient = K.gradients(loss, genTensor)
    outputs = [loss]
    if isinstance(gradient, (list, tuple)):
        outputs += gradient
    else:
        outputs.append(gradient)
    global f_outputs
    f_outputs = K.function([genTensor], outputs)
    print("   Beginning transfer.")
    evaluator = Evaluator()
    for i in range(TRANSFER_ROUNDS):
        index = i + 1
        print("   Step %d." % i)
        tData, tLoss, info = fmin_l_bfgs_b(evaluator.loss, tData.flatten(), fprime=evaluator.grads, maxfun=20)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(tData.copy())
        saveFile = "image" + str(index) + ".jpg"
        imageio.imwrite(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")


def eval_loss_and_grads(img):
    img = img.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    outs = f_outputs([img])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# =========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])  # Content image.
    sData = preprocessData(raw[1])  # Style image.
    tData = preprocessData(raw[2])  # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
