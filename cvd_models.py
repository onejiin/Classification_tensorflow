# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from network import *

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    net = slim.conv2d(model_input, 8, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
    net = slim.max_pool2d(net,[2, 2], scope='pool1')
    net = slim.conv2d(net, 16, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.conv2d(net, 8, [1, 1], normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

    net = slim.flatten(net)
    net = slim.fully_connected(
        net, 8, activation_fn=tf.nn.relu, scope='fc1')
    net = slim.dropout(net, 0.5, scope='dropout')
    output = slim.fully_connected(
        net, num_classes-1, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(0.0005), scope='fc3')
    return {"predictions": output}


class FineTuningModel(models.BaseModel):
  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
    # TODO weight decay loss tern
    # Layer 1 (conv-relu-pool-lrn)
    #keep_rate = tf.placeholder(tf.float32)
    batch_size = 100
    model_input = tf.reshape(model_input, (batch_size, 227, 227, 3))

    conv1 = conv(model_input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    # Layer 2 (conv-relu-pool-lrn)
    conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
    conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
    norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    # Layer 3 (conv-relu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
    # Layer 4 (conv-relu)
    conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
    # Layer 5 (conv-relu-pool)
    conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
    # Layer 6 (fc-relu-drop)
    fc6 = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(fc6, 6 * 6 * 256, 4096, name='fc6')
#    fc6 = dropout(fc6, keep_rate)
    # Layer 7 (fc-relu-drop)
    fc7 = fc(fc6, 4096, 4096, name='fc7')
 #   fc7 = dropout(fc7, keep_rate)
    # Layer 8 (fc-prob)
    fc8 = fc(fc7, 4096, 1, relu=False, name='fc8')

    return {"predictions": fc8}
    #return fc8

