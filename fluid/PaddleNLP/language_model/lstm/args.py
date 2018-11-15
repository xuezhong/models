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

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="Specify the path to load trained models.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The sequence number of a mini-batch data. (default: %(default)d)")
    parser.add_argument(
        "--embed_size",
        type=int,
        default=300,
        help="The dimension of embedding table. (default: %(default)d)")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="The size of rnn hidden unit. (default: %(default)d)")
    parser.add_argument(
        "--init_scale",
        type=float,
        default=0.01,
        help="init scale. (default: %(default)f)")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="The size of rnn layers. (default: %(default)d)")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="The size of sequence len. (default: %(default)d)")
    parser.add_argument(
        "--data_path", type=str, help="all the data for train,valid,test")
    parser.add_argument("--vocab_path", type=str, help="vocab file path")
    parser.add_argument('--para_init', action='store_true')
    parser.add_argument(
        '--use_gpu', type=bool, default=False, help='whether using gpu')
    parser.add_argument(
        '--log_path',
        help='path of the log file. If not set, logs are printed to console')
    parser.add_argument('--enable_ce', action='store_true')
    parser.add_argument('--optim', default='adam', help='optimizer type')
    parser.add_argument('--para_print', action='store_true')
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate used to train the model. (default: %(default)f)")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="log the train loss every n batches."
        "(default: %(default)d)")
    parser.add_argument(
        "--dev_interval",
        type=int,
        default=1000,
        help="cal dev loss every n batches."
        "(default: %(default)d)")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--max_epoch', type=float, default=10)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    return args
