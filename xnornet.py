# Copyright 2021 Loro Francesco
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Francesco Loro"
__email__ = "francesco.official@gmail.com"
__supervisor__ = "Danilo Pau"
__email__ = "danilo.pau@st.com"

# Download pretrained weight from:
# Xnornet -> https://drive.google.com/file/d/1-rDz7bFE0d9LmF1A0RfOfGIguuuuErPp/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_XNORNET = "weights/xnornet_weights.h5"
XNORNET_NAME = "xnorNet"


class XnorNet:
  """Class to create and load weights of: xnornet

  Attributes:
        network_name: Name of the network
  """

  def __init__(self):
    self.__weights_path = PATH_XNORNET
    self.network_name = XNORNET_NAME

  def build(self):
    """Build the model

    Returns:
      Qkeras and larq models
    """
    qkeras_network = self.build_qkeras_xnornet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_xnornet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  @staticmethod
  def add_qkeras_quant_block(given_model, filters_num, kernel, bn,
                             padding="same"):
    """Adds a sequence of layers to the given model

    Add a sequence of: Activation quantization, Quantized Conv2D, and
    BatchNormalization if bool bn is True to the given model

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D
      kernel: kernel size for Conv2D
      bn: boolean to decide if BatchNorm is performed or not

    Returns:
      Given Model plus the sequence
    """
    x = q.QActivation("binary(alpha=1)")(given_model)
    x = q.QConv2D(filters_num, kernel, padding=padding,
                  kernel_quantizer="binary(alpha=1)", use_bias=False,
                  kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    if bn:
      x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                             epsilon=1e-4)(x)
    return x

  @staticmethod
  def add_larq_quant_block(given_model, filters_num, kernel, bn,
                           padding="same"):
    """Same method of add_qkeras_conv_block but for a larq network
    """
    x = lq.layers.QuantConv2D(filters_num, kernel, padding=padding,
                              kernel_quantizer="ste_sign",
                              input_quantizer="ste_sign",
                              kernel_constraint="weight_clip",
                              use_bias=False,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                5e-7))(given_model)
    if bn:
      x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                             epsilon=1e-4)(x)
    return x

  def build_qkeras_xnornet(self):
    """Build the qkeras version of the xnornet

    Return:
      Qkeras model of the xnornet
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    qkeras_xnornet = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4),
                                            padding="same",
                                            use_bias=False,
                                            kernel_regularizer=
                                            tf.keras.regularizers.l2(
                                              5e-7),
                                            )(input_layer)

    qkeras_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        scale=False,
                                                        epsilon=1e-5)(
      qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.Activation("relu")(qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                            strides=(2, 2))(
      qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        scale=False,
                                                        epsilon=1e-4)(
      qkeras_xnornet)
    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=256,
                                                 kernel=(5, 5), bn=False)
    qkeras_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                            strides=(2, 2))(
      qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        scale=False,
                                                        epsilon=1e-4)(
      qkeras_xnornet)

    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=384,
                                                 kernel=(3, 3), bn=True)
    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=384,
                                                 kernel=(3, 3), bn=True)

    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=256,
                                                 kernel=(3, 3), bn=False)
    qkeras_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                            strides=(2, 2))(
      qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        scale=False,
                                                        epsilon=1e-4)(
      qkeras_xnornet)

    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=4096,
                                                 kernel=(6, 6), bn=True,
                                                 padding="valid")

    qkeras_xnornet = self.add_qkeras_quant_block(qkeras_xnornet,
                                                 filters_num=4096,
                                                 kernel=(1, 1), bn=True,
                                                 padding="valid")

    qkeras_xnornet = tf.keras.layers.Activation("relu")(qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.Flatten()(qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.Dense(
      1000,
      use_bias=False,
      kernel_regularizer=tf.keras.regularizers.l2(5e-7)
    )(qkeras_xnornet)
    qkeras_xnornet = tf.keras.layers.Activation("softmax", dtype="float32")(
      qkeras_xnornet)
    qkeras_xnornet = tf.keras.models.Model(
      inputs=input_layer, outputs=qkeras_xnornet, name="xnornet"
    )
    qkeras_xnornet.load_weights(self.__weights_path)
    return qkeras_xnornet

  def build_larq_xnornet(self):
    """Build the larq version of the xnornet

    Return:
      Larq model of the xnornet
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    larq_xnornet = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4),
                                          padding="same",
                                          use_bias=False,
                                          kernel_regularizer=
                                          tf.keras.regularizers.l2(
                                            5e-7),
                                          )(input_layer)

    larq_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                                      epsilon=1e-5)(
      larq_xnornet)
    larq_xnornet = tf.keras.layers.Activation("relu")(larq_xnornet)
    larq_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                        strides=(2, 2))(
      larq_xnornet)
    larq_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                                      epsilon=1e-4)(
      larq_xnornet)

    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=256,
                                             kernel=(5, 5), bn=False)
    larq_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                        strides=(2, 2))(
      larq_xnornet)
    larq_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                                      epsilon=1e-4)(
      larq_xnornet)

    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=384,
                                             kernel=(3, 3), bn=True)
    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=384,
                                             kernel=(3, 3), bn=True)

    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=256,
                                             kernel=(3, 3), bn=False)
    larq_xnornet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                        strides=(2, 2))(
      larq_xnornet)
    larq_xnornet = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False,
                                                      epsilon=1e-4)(
      larq_xnornet)

    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=4096,
                                             kernel=(6, 6), bn=True,
                                             padding="valid")

    larq_xnornet = self.add_larq_quant_block(larq_xnornet, filters_num=4096,
                                             kernel=(1, 1), bn=True,
                                             padding="valid")

    larq_xnornet = tf.keras.layers.Activation("relu")(larq_xnornet)
    larq_xnornet = tf.keras.layers.Flatten()(larq_xnornet)
    larq_xnornet = tf.keras.layers.Dense(
      1000,
      use_bias=False,
      kernel_regularizer=tf.keras.regularizers.l2(5e-7)
    )(larq_xnornet)
    larq_xnornet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_xnornet)
    larq_xnornet = tf.keras.models.Model(
      inputs=input_layer, outputs=larq_xnornet, name="xnornet"
    )
    larq_xnornet.load_weights(self.__weights_path)
    return larq_xnornet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network = XnorNet()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=XNORNET_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=XNORNET_NAME)
