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

import numpy as np
import time
import os
import random

import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from args import *
import lm_model
import logging
import pickle

SEED = 123


def prepare_batch_input(batch, epoch_id=0, with_lr=True):
    x, y = batch
    inst = []
    for i in range(len(x)):
        inst.append([x[i], y[i]])
    return inst


def batch_reader(batch_list, args):
    res = []
    for batch in batch_list:
        res.append(prepare_batch_input(batch, args))
    return res


def read_multiple(reader, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        res = []
        for item in reader():
            res.append(item)
            if len(res) == count:
                yield res
                res = []
        if len(res) == count:
            yield res
        elif not clip_last:
            data = []
            for item in res:
                data += item
            if len(data) > count:
                inst_num_per_part = len(data) // count
                yield [
                    data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                    for i in range(count)
                ]

    return __impl__


def LodTensor_Array(lod_tensor):
    lod = lod_tensor.lod()
    array = np.array(lod_tensor)
    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array


def print_para(train_prog, train_exe, logger, args):
    if args.para_print:
        param_list = train_prog.block(0).all_parameters()
        param_name_list = [p.name for p in param_list]
        num_sum = 0
        for p_name in param_name_list:
            p_array = np.array(train_exe.scope.find_var(p_name).get_tensor())
            param_num = np.prod(p_array.shape)
            num_sum = num_sum + param_num
            logger.info(
                "param: {0},  mean={1}  max={2}  min={3}  num={4} {5}".format(
                    p_name,
                    p_array.mean(),
                    p_array.max(), p_array.min(), p_array.shape, param_num))
        logger.info("total param num: {0}".format(num_sum))


def get_current_model_para(train_prog, train_exe):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    return vals


def save_para_npz(train_prog, train_exe):
    logger.info("begin to save model to model_base")
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    emb = vals["embedding_para"]
    logger.info("begin to save model to model_base")
    np.savez("mode_base", **vals)


def prepare_input(batch, epoch_id=0, with_lr=True):
    x, y = batch
    inst = []
    for i in range(len(x)):
        inst.append([x[i], y[i]])
    return inst


def eval(dev_data, inference_program, feed_order, dev_count, loss, place,
         logger, args):
    parallel_executor = fluid.ParallelExecutor(
        main_program=inference_program, use_cuda=bool(args.use_gpu))
    print_para(inference_program, parallel_executor, logger, args)

    # Use test set as validation each pass
    total_loss = 0.0
    n_batch_cnt = 0
    n_batch_loss = 0.0
    val_feed_list = [
        inference_program.global_block().var(var_name)
        for var_name in feed_order
    ]
    val_feeder = fluid.DataFeeder(val_feed_list, place)
    dev_data_iter = reader.get_data_iter(dev_data, args.batch_size,
                                         args.num_steps)
    dev_reader = read_multiple(dev_data_iter, dev_count)

    for batch_id, batch_list in enumerate(dev_reader(), 1):
        feed_data = batch_reader(batch_list, args)
        val_fetch_outs = parallel_executor.run(
            feed=list(val_feeder.feed_parallel(feed_data, dev_count)),
            fetch_list=[loss.name],
            return_numpy=False)
        total_loss += np.array(val_fetch_outs[0]).sum()

        n_batch_cnt += len(np.array(val_fetch_outs[0]))
        n_batch_loss += np.array(val_fetch_outs[0]).sum()
        log_every_n_batch = args.log_interval
        if log_every_n_batch > 0 and batch_id % log_every_n_batch == 0:
            logger.info('Average dev loss from batch {} to {} is {}'.format(
                batch_id - log_every_n_batch + 1, batch_id, "%.10f" % (
                    n_batch_loss / n_batch_cnt)))
            n_batch_loss = 0.0
            n_batch_cnt = 0
        batch_offset = 0

    ppl = np.exp(total_loss / n_batch_cnt)
    return ppl


def train():
    args = parse_args()
    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.enable_ce:
        fluid.default_startup_program().random_seed = SEED
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    vocab_size = 1408482
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    data_path = args.data_path
    logger.info("begin to load data")
    raw_data = reader.ptb_raw_data(data_path, args.vocab_path)
    logger.info("finished load data")
    train_data, valid_data, test_data, vocab_size = raw_data

    logger.info('Initialize the model...')

    if not args.use_gpu:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    # Training process
    loss, feed_order = lm_model.lm_model(
        hidden_size,
        vocab_size,
        batch_size,
        num_layers=args.num_layers,
        num_steps=args.num_steps,
        init_scale=args.init_scale,
        args=args)
    # clone from default main program and use it as the validation program
    # build model
    main_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_program, startup_prog):
        with fluid.unique_name.guard():
            # Training process
            loss, feed_order = lm_model.lm_model(
                hidden_size,
                vocab_size,
                batch_size,
                num_layers=args.num_layers,
                num_steps=args.num_steps,
                init_scale=args.init_scale,
                args=args)
            inference_program = main_program.clone(for_test=True)

            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=args.max_grad_norm))

            # build optimizer
            if args.optim == 'sgd':
                optimizer = fluid.optimizer.SGD(
                    learning_rate=args.learning_rate)
            elif args.optim == 'adam':
                optimizer = fluid.optimizer.Adam(
                    learning_rate=args.learning_rate)
            elif args.optim == 'rprop':
                optimizer = fluid.optimizer.RMSPropOptimizer(
                    learning_rate=args.learning_rate)
            else:
                logger.error('Unsupported optimizer: {}'.format(args.optim))
                exit(-1)
            optimizer.minimize(loss)

            # initialize parameters
            place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
            exe = Executor(place)
            if args.load_dir:
                logger.info('load from {}'.format(args.load_dir))
                fluid.io.load_persistables(
                    exe, args.load_dir, main_program=main_program)
            else:
                exe.run(startup_prog)

            # prepare data
            feed_list = [
                main_program.global_block().var(var_name)
                for var_name in feed_order
            ]
            feeder = fluid.DataFeeder(feed_list, place)

            logger.info('Training the model...')
            parallel_executor = fluid.ParallelExecutor(
                main_program=main_program, use_cuda=bool(args.use_gpu))
            print_para(main_program, parallel_executor, logger, args)

            # get train epoch size
            log_interval = args.log_interval
            total_time = 0.0
            for epoch_id in range(args.max_epoch):
                start_time = time.time()
                logger.info("epoch id {}".format(epoch_id))
                train_data_iter = reader.get_data_iter(train_data, batch_size,
                                                       args.num_steps)
                train_reader = read_multiple(train_data_iter, dev_count)

                total_loss = 0
                total_num = 0
                n_batch_loss = 0.0
                for batch_id, batch_list in enumerate(train_reader()):
                    feed_data = batch_reader(batch_list, args)
                    fetch_outs = parallel_executor.run(
                        feed=list(feeder.feed_parallel(feed_data, dev_count)),
                        fetch_list=[loss.name],
                        return_numpy=False)
                    cost_train = np.array(fetch_outs[0]).mean()
                    total_num += args.batch_size * dev_count
                    n_batch_loss += cost_train
                    total_loss += cost_train * args.batch_size * dev_count

                    if batch_id > 0 and batch_id % log_interval == 0:
                        ppl = np.exp(total_loss / total_num)
                        logger.info("ppl {} {} ".format(batch_id, ppl))
                    if batch_id > 0 and batch_id % args.dev_interval == 0:
                        valid_ppl = eval(valid_data, inference_program,
                                         feed_order, dev_count, loss, place,
                                         logger, args)
                        logger.info("valid ppl {}".format(valid_ppl))

                end_time = time.time()
                total_time += end_time - start_time
                logger.info("train ppl {}".format(ppl))

                if epoch_id == max_epoch - 1 and args.enable_ce:
                    logger.info("lstm_language_model_duration\t%s" %
                                (total_time / max_epoch))
                    logger.info("lstm_language_model_loss\t%s" % ppl[0])

                model_path = os.path.join("model_new/", str(epoch_id))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                fluid.io.save_persistables(
                    executor=exe, dirname=model_path, main_program=main_program)
                valid_ppl = eval(valid_data, inference_program, feed_order,
                                 dev_count, loss, place, logger, args)
                logger.info("valid ppl {}".format(valid_ppl))
            test_ppl = eval(test_data, inference_program, feed_order, dev_count,
                            loss, place, logger, args)
            logger.info("test ppl {}".format(test_ppl))


if __name__ == '__main__':
    train()
