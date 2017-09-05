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

