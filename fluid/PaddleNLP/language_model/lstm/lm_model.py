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


def dropout(input, args):
    if args.dropout:
        return layers.dropout(
            input,
            dropout_prob=args.dropout,
            seed=args.random_seed,
            is_test=False)
    else:
        return input


def lstmp_encoder(input_seq, gate_size, para_name, proj_size, args):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    if args.para_init:
        init = fluid.initializer.Constant(args.init1)
        init_b = fluid.initializer.Constant(0.0)
    else:
        init = None
        init_b = None
    input_seq = dropout(input_seq, args)
    input_forward_proj = layers.fc(input=input_seq,
                                   param_attr=fluid.ParamAttr(
                                       name=para_name + '_gate_w',
                                       initializer=init),
                                   size=gate_size * 4,
                                   act=None,
                                   bias_attr=False)
    forward, _ = layers.dynamic_lstmp(
        input=input_forward_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        use_peepholes=False,
        param_attr=fluid.ParamAttr(initializer=init),
        bias_attr=fluid.ParamAttr(initializer=init_b))

    encoder_out = forward
    return encoder_out


def lm_model(hidden_size,
             vocab_size,
             batch_size,
             num_layers=2,
             num_steps=20,
             init_scale=0.1,
             args=None):

    x_f = layers.data(name="x", shape=[1], dtype='int64', lod_level=1)
    y_f = layers.data(name="y", shape=[1], dtype='int64', lod_level=1)

    x_b = layers.data(name="x_r", shape=[1], dtype='int64', lod_level=1)
    y_b = layers.data(name="y_r", shape=[1], dtype='int64', lod_level=1)

    emb_size = args.embed_size
    lstm_outputs = []

    def encoder(x, y, para_name, args):
        x_emb = layers.embedding(
            input=x,
            size=[vocab_size, emb_size],
            dtype='float32',
            is_sparse=True,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        rnn_out = lstmp_encoder(x_emb, hidden_size, para_name + 'layer1',
                                emb_size, args)
        rnn_out2 = lstmp_encoder(rnn_out, hidden_size, para_name + 'layer2',
                                 emb_size, args)

        rnn_out2 = rnn_out2 + rnn_out
        rnn_out2 = dropout(rnn_out2, args)

        softmax_weight = layers.create_parameter([vocab_size, emb_size], dtype="float32", name="softmax_weight", \
                default_initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale))
        softmax_bias = layers.create_parameter([vocab_size], dtype="float32", name='softmax_bias', \
                default_initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale))
        projection = layers.matmul(rnn_out2, softmax_weight, transpose_y=True)
        projection = layers.elementwise_add(projection, softmax_bias)

        projection = layers.reshape(projection, shape=[-1, vocab_size])

        loss = layers.softmax_with_cross_entropy(
            logits=projection, label=y, soft_label=False)
        return [x_emb, rnn_out, rnn_out2, projection, loss]

    forward = encoder(x_f, y_f, 'fw_', args)
    backward = encoder(x_b, y_b, 'bw_', args)

    losses = layers.concat([forward[-1], backward[-1]])
    loss = layers.reduce_mean(losses)
    loss.permissions = True

    if args.debug:
        layers.Print(y, summarize=10)
        layers.Print(x_emb, summarize=10)
        layers.Print(projection, summarize=10)
        layers.Print(loss, summarize=10)
    grad_vars = [x_f, y_f, x_b, y_b] + forward + backward
    grad_vars_name = [
        'x', 'y', 'x_r', 'y_r', 'x_emb', 'rnn_out', 'rnn_out2', 'proj', 'loss',
        'x_emb_r', 'rnn_out_r', 'rnn_out2_r', 'proj_r', 'loss_r'
    ]
    feeding_list = ['x', 'y', 'x_r', 'y_r']
    return loss, feeding_list, grad_vars, grad_vars_name
