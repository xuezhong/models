#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np


def linear(inputs, para_name, size):
    return layers.fc(input=inputs,
                     size=size,
                     param_attr=fluid.ParamAttr(name=para_name + '_w'),
                     bias_attr=fluid.ParamAttr(name=para_name + '_b'))


def bi_lstm_encoder(input_seq, gate_size, para_name, proj_size):
    # A bi-directional lstm encoder implementation.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    input_forward_proj = layers.fc(
        input=input_seq,
        param_attr=fluid.ParamAttr(name=para_name + '_fw_gate_w'),
        size=gate_size * 4,
        act=None,
        bias_attr=False)
    input_reversed_proj = layers.fc(
        input=input_seq,
        param_attr=fluid.ParamAttr(name=para_name + '_bw_gate_w'),
        size=gate_size * 4,
        act=None,
        bias_attr=False)
    forward, _ = layers.dynamic_lstmp(
        input=input_forward_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        use_peepholes=False)  #,
    #param_attr=fluid.ParamAttr(name=para_name + '_fw_lstm_w'),
    #bias_attr=fluid.ParamAttr(name=para_name + '_fw_lstm_b'))
    reversed, _ = layers.dynamic_lstmp(
        input=input_reversed_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        is_reverse=True,
        use_peepholes=False)
    #param_attr=fluid.ParamAttr(name=para_name + '_bw_lstm_w'),
    #bias_attr=fluid.ParamAttr(name=para_name + '_bw_lstm_b'))

    encoder_out = [forward, reversed]
    return encoder_out


def lm_model(hidden_size,
             vocab_size,
             batch_size,
             num_layers=2,
             num_steps=20,
             init_scale=0.1,
             args=None):

    x = layers.data(name="x", shape=[1], dtype='int64', lod_level=1)
    y = layers.data(name="y", shape=[1], dtype='int64', lod_level=1)

    emb_size = args.embed_size
    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, emb_size],
        dtype='float32',
        is_sparse=True,
        param_attr=fluid.ParamAttr(
            name='embedding_para',
            initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale)))
    dropout = args.dropout
    if dropout != None and dropout > 0.0:
        x_emb = layers.dropout(
            x_emb,
            dropout_prob=dropout,
            dropout_implementation='upscale_in_train')
    rnn_out = bi_lstm_encoder(x_emb, hidden_size, 'layer1', emb_size)
    lstm_outputs = bi_lstm_encoder(rnn_out, hidden_size, 'layer2', emb_size)

    softmax_weight = layers.create_parameter([emb_size, vocab_size], dtype="float32", name="softmax_weight", \
            default_initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale))
    softmax_bias = layers.create_parameter([vocab_size], dtype="float32", name='softmax_bias', \
            default_initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale))
    losses = []
    for lstm_output in lstm_outputs:
        projection = layers.matmul(lstm_output, softmax_weight)
        projection = layers.elementwise_add(projection, softmax_bias)

        projection = layers.reshape(projection, shape=[-1, vocab_size])

        one_loss = layers.softmax_with_cross_entropy(
            logits=projection, label=y, soft_label=False)
        losses.append(one_loss)
    losses = layers.concat(losses)
    loss = layers.reduce_mean(losses)
    loss.permissions = True

    if args.debug:
        layers.Print(y, summarize=10)
        layers.Print(x_emb, summarize=10)
        layers.Print(projection, summarize=10)
        layers.Print(loss, summarize=10)

    feeding_list = ['x', 'y']
    return loss, feeding_list
