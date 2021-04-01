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
# b2rnet -> https://drive.google.com/file/d/1-rDz7bFE0d9LmF1A0RfOfGIguuuuErPp/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_REAL2BINARY = "weights/r2b_weights.h5"
REAL2BINARY_NAME = "real_to_binary"


class R2BNet:
  """Class to create and load weights of: real_to_binary

  Attributes:
        network_name: Name of the network
        filters: Number of filters for Conv2D
  """

  def __init__(self):
    self.__weights_path = PATH_REAL2BINARY
    self.network_name = REAL2BINARY_NAME
    self.__filter = 32
    self.__reduction_neuron = 4
    self.__expand_neuron = 32
    self.__pool_size = 112
    self.__repetitions = (4, 3, 3)

  @staticmethod
  def add_larq_scale_block(given_model, filters_num, reduct_neurons,
                           exp_neurons, pool_size, strides=1):


    conv_input = tf.keras.layers.BatchNormalization(momentum=0.99)(given_model)
    conv_output = lq.layers.QuantConv2D(filters_num, kernel_size=3,
                                        strides=strides,
                                        padding="same",
                                        input_quantizer="ste_sign",
                                        kernel_quantizer="ste_sign",
                                        kernel_constraint="weight_clip",
                                        kernel_initializer="glorot_normal",
                                        use_bias=False)(conv_input)
    scales = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(conv_input)
    scales = tf.keras.layers.Flatten()(scales)
    scales = tf.keras.layers.Dense(reduct_neurons, activation="relu",
                                   kernel_initializer="he_normal",
                                   use_bias=False,
                                   )(scales)
    scales = tf.keras.layers.Dense(exp_neurons, activation="sigmoid",
                                   kernel_initializer="he_normal",
                                   use_bias=False)(scales)
    scales = tf.keras.layers.Reshape((1, 1, exp_neurons))(scales)

    x = tf.keras.layers.Multiply()([conv_output, scales])
    return tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

  @staticmethod
  def add_transition_block(given_model, filters_num, scaling_filters_num,
                           reduct_neurons, exp_neurons, pool_size):
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(given_model)
    x = tf.keras.layers.Conv2D(filters_num, kernel_size=1, strides=1,
                               kernel_initializer="glorot_normal",
                               padding="valid", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    scales = R2BNet.add_larq_scale_block(given_model, scaling_filters_num,
                                              reduct_neurons, exp_neurons,
                                              strides=2, pool_size=pool_size)

    return tf.keras.layers.Add()([scales, x])

  def build(self):
    """Build the model

    Returns:
      qkeras and larq models
    """
    qkeras_network = self.build_qkeras_r2bnet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_r2bnet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_r2bnet(self):
    """Build the qkeras version of the real_to_binarynet

    Returns:
      Qkeras model of the real_to_binarynet

    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    qkeras_r2bnet = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                           kernel_initializer="glorot_normal",
                                           padding="same", use_bias=False)(
      input_layer)
    qkeras_r2bnet = tf.keras.layers.BatchNormalization(momentum=0.99)(
      qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.layers.PReLU(shared_axes=[1, 2])(qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(
      qkeras_r2bnet)

    for scale, reps in enumerate(self.__repetitions, 1):
      for _ in range(0, reps):
        scaled = \
          self.add_larq_scale_block(given_model=qkeras_r2bnet,
                                    filters_num=(self.__filter * 2 ** scale),
                                    reduct_neurons=
                                    (self.__reduction_neuron * 2 ** scale),
                                    exp_neurons=
                                    (self.__expand_neuron * 2 ** scale),
                                    pool_size=
                                    (self.__pool_size // 2 ** scale))
        qkeras_r2bnet = tf.keras.layers.Add()([qkeras_r2bnet, scaled])
      qkeras_r2bnet = \
        self.add_transition_block(given_model=qkeras_r2bnet,
                                  filters_num=(self.__filter * 2 ** scale) * 2,
                                  scaling_filters_num=
                                  (self.__filter * 2 ** scale) * 2,
                                  reduct_neurons=
                                  (self.__reduction_neuron * 2 ** scale),
                                  exp_neurons=
                                  (self.__expand_neuron * 2 ** scale) * 2,
                                  pool_size=(self.__pool_size // 2 ** scale))
    for _ in range(0, 3):
      scales = self.add_larq_scale_block(qkeras_r2bnet, filters_num=512,
                                         reduct_neurons=64, exp_neurons=512,
                                         pool_size=7)
      qkeras_r2bnet = tf.keras.layers.Add()([qkeras_r2bnet, scales])

    qkeras_r2bnet = tf.keras.layers.AveragePooling2D(pool_size=7)(qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.layers.Flatten()(qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.layers.Dense(1000)(qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.layers.Activation("softmax", dtype="float32")(
      qkeras_r2bnet)
    qkeras_r2bnet = tf.keras.Model(inputs=input_layer, outputs=qkeras_r2bnet)
    qkeras_r2bnet.load_weights(self.__weights_path)
    return qkeras_r2bnet

  def build_larq_r2bnet(self):
    """Build the larq version of the real_to_binarynet

    Returns:
      Larq model of the real_to_binarynet

    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    larq_r2bnet = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                         kernel_initializer="glorot_normal",
                                         padding="same", use_bias=False)(
      input_layer)
    larq_r2bnet = tf.keras.layers.BatchNormalization(momentum=0.99)(larq_r2bnet)
    larq_r2bnet = tf.keras.layers.PReLU(shared_axes=[1, 2])(larq_r2bnet)
    larq_r2bnet = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(
      larq_r2bnet)

    for scale, reps in enumerate(self.__repetitions, 1):
      for _ in range(0, reps):
        scaled = \
          self.add_larq_scale_block(given_model=larq_r2bnet,
                                    filters_num=(self.__filter * 2 ** scale),
                                    reduct_neurons=
                                    (self.__reduction_neuron * 2 ** scale),
                                    exp_neurons=
                                    (self.__expand_neuron * 2 ** scale),
                                    pool_size=(self.__pool_size // 2 ** scale))
        larq_r2bnet = tf.keras.layers.Add()([larq_r2bnet, scaled])
      larq_r2bnet = \
        self.add_transition_block(given_model=larq_r2bnet,
                                  filters_num=
                                  (self.__filter * 2 ** scale) * 2,
                                  scaling_filters_num=
                                  (self.__filter * 2 ** scale) * 2,
                                  reduct_neurons=
                                  (self.__reduction_neuron * 2 ** scale),
                                  exp_neurons=
                                  (self.__expand_neuron * 2 ** scale) * 2,
                                  pool_size=(self.__pool_size // 2 ** scale))
    for _ in range(0, 3):
      scales = self.add_larq_scale_block(larq_r2bnet, filters_num=512,
                                         reduct_neurons=64, exp_neurons=512,
                                         pool_size=7)
      larq_r2bnet = tf.keras.layers.Add()([larq_r2bnet, scales])

    larq_r2bnet = tf.keras.layers.AveragePooling2D(pool_size=7)(larq_r2bnet)
    larq_r2bnet = tf.keras.layers.Flatten()(larq_r2bnet)
    larq_r2bnet = tf.keras.layers.Dense(1000)(larq_r2bnet)
    larq_r2bnet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_r2bnet)
    larq_r2bnet = tf.keras.Model(inputs=input_layer, outputs=larq_r2bnet)
    larq_r2bnet.load_weights(self.__weights_path)
    return larq_r2bnet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(10)

  network = R2BNet()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=REAL2BINARY_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=REAL2BINARY_NAME)
