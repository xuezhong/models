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

import data
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

names = []
grad_names = []
import collections

name_dict = collections.OrderedDict()
name_dict['embedding_para'] = 1
#'''
name_dict['lstmp_0.b_0'] = 21
name_dict['fw_layer1_gate_w'] = 22
name_dict['lstmp_0.w_0'] = 22
name_dict['lstmp_0.w_1'] = 23
name_dict['lstmp_1.b_0'] = 31
name_dict['bw_layer1_gate_w'] = 32
name_dict['lstmp_1.w_0'] = 32
name_dict['lstmp_1.w_1'] = 33
name_dict['lstmp_2.b_0'] = 41
name_dict['fw_layer2_gate_w'] = 42
name_dict['lstmp_2.w_0'] = 42
name_dict['lstmp_2.w_1'] = 43
name_dict['lstmp_3.b_0'] = 51
name_dict['bw_layer2_gate_w'] = 52
name_dict['lstmp_3.w_0'] = 52
name_dict['lstmp_3.w_1'] = 53
#'''
name_dict['softmax_weight'] = 62
name_dict['softmax_bias'] = 61

slot_dict = {}


def init_slot():
    global slot_dict
    slot_dict = {}


def name2slot(para_name, exact=False):
    res = []
    if exact:
        if para_name in name_dict:
            return [name_dict[para_name]]
        else:
            return []
    for key_name in name_dict.keys():
        if para_name.find(key_name) >= 0:
            res.append(name_dict[key_name])
    return res


def update_slot(slots, p_array):
    p_mean, p_max, p_min, p_num = p_array.mean(), p_array.max(), p_array.min(
    ), np.prod(p_array.shape)
    for slot in slots:
        if slot in slot_dict:
            s_mean, s_max, s_min, s_num = slot_dict[slot]
            s_mean = (s_mean * s_num + p_mean * p_num) / (p_num + s_num)
            s_max = max(s_max, p_max)
            s_min = min(s_min, p_min)
            s_num = p_num + s_num
            slot_dict[slot] = [s_mean, s_max, s_min, s_num]
        else:
            slot_dict[slot] = [p_mean, p_max, p_min, p_num]


def record_slot(logger):
    for slot in slot_dict:
        logger.info("slot:" + "\t".join(
            [str(x) for x in [slot] + slot_dict[slot]]))


def var_print(tag, p_array, p_name, name, detail, logger):
    param_num = np.prod(p_array.shape)
    p_array3 = np.multiply(np.multiply(p_array, p_array), p_array)
    logger.info(
        tag +
        ": {0} ({1}),  l3={2} sum={3}  max={4}  min={5} mean={6} num={7} {8}".
        format(p_name, name,
               p_array3.sum(),
               p_array.sum(),
               p_array.max(),
               p_array.min(), p_array.mean(), p_array.shape, param_num))
    if detail:
        logger.info(" ".join([
            tag + "[", p_name, '] shape [', str(p_array.shape), ']', str(
                p_array)
        ]))


def save_var(p_array, name, logger, args):
    if args.save_para_path:
        if name2slot(name, exact=True):
            name = 'slot_' + str(name2slot(name, exact=True)[0])
        else:
            name = name.replace('/', '%')
        with open(os.path.join(args.save_para_path, name + '.data'),
                  'wb') as fout:
            pickle.dump(p_array, fout)


def save_para(train_prog, train_exe, logger, args=None):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for var in param_list:
        p_name = var.name
        p_array = np.array(train_exe.scope.find_var(p_name).get_tensor(
        )).astype('float64')
        save_var(p_array, p_name, logger, args)


def load_var(tensor, slot, place, logger, args):
    with open(
            os.path.join(args.para_load_dir, 'slot_' + str(slot[0]) + '.data'),
            'rb') as fin:
        p_array = pickle.load(fin)
        if slot in [22, 32, 42, 52]:
            tensor.set(p_array.astype(np.float32), place)
        else:
            tensor.set(p_array.astype(np.float32), place)


def listDir(rootDir):
    res = []
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            res.append(pathname)
    return res


#load from slot file
def load_params(train_prog, train_exe, place, logger, args=None):
    if not args.para_load_dir:
        return
    logger.info('loading para from {}'.format(args.para_load_dir))
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for data in listDir(args.para_load_dir):
        slot = int(data.split('_')[1].split('.')[0])
        with open(data, 'rb') as fin:
            p_array = pickle.load(fin)
            p_array = p_array.reshape((-1))
            offset = 0
            for name in name_dict:
                s = name_dict[name]
                if s == slot:
                    tensor = train_exe.scope.find_var(name).get_tensor()
                    shape = tensor.shape()
                    tensor_len = np.prod(shape)
                    new_array = p_array[offset:offset + tensor_len]
                    new_array = new_array.reshape(shape)
                    tensor.set(new_array.astype(np.float32), place)
                    logger.info('loaded {}[{}] from {}[{}:{}]'.format(
                        name, shape, data, offset, offset + tensor_len))
                    offset += tensor_len


def load_para(train_prog, train_exe, place, logger, args=None):
    if not args.para_load_dir:
        return
    logger.info('loading para form {}'.format(args.para_load_dir))
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for var in param_list:
        p_name = var.name
        tensor = train_exe.scope.find_var(p_name).get_tensor()
        if name2slot(var.name, exact=True):
            slot = name2slot(var.name, exact=True)
            load_var(tensor, slot, place, logger, args)


names = []
grad_names = [
]  #[['create_parameter_0.w_0@GRAD', 'create_parameter_0.w_0']]#,['embedding_para@GRAD', 'embedding_para']]


def debug_init(train_prog, vars, vars_name):
    for i in range(len(vars)):
        name = vars[i].name + '@GRAD'
        grad_names.append([name, vars_name[i]])
        name = vars[i].name
        names.append([name, vars_name[i]])
    for name in names:
        train_prog.block(0).var(name[0]).persistable = True
    for name in grad_names:
        if train_prog.block(0).has_var(name[0]):
            train_prog.block(0).var(name[0]).persistable = True

    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for p_name in param_name_list:
        p_name = p_name + '@GRAD'
        if train_prog.block(0).has_var(p_name):
            train_prog.block(0).var(p_name).persistable = True


def debug_print(train_exe, logger, args):
    if not args.para_print:
        return
    for name_pair in names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not train_exe.scope.find_var(name):
            logger.info("var: {0} not find".format(p_name))
            continue
        p_array = np.array(train_exe.scope.find_var(name).get_tensor()).astype(
            'float64')
        var_print('var', p_array, p_name, name, args.detail, logger)
    for name_pair in grad_names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not train_exe.scope.find_var(name):
            logger.info("grad: {0} not find".format(p_name))
            continue
        p_array = np.array(train_exe.scope.find_var(name).get_tensor()).astype(
            'float64')
        var_print('grad', p_array, p_name, name, args.detail, logger)


def print_para(train_prog, train_exe, logger, args):
    if not args.para_print:
        return
    debug_print(train_exe, logger, args)
    init_slot()
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    num_sum = 0
    for p_name in param_name_list:
        p_name = p_name + '@GRAD'
        if not train_exe.scope.find_var(p_name):
            logger.info("grad para: {0} not find".format(p_name))
            continue
        try:
            p_array = np.array(train_exe.scope.find_var(p_name).get_tensor())
        except:
            logger.info("grad para: {0} failed".format(p_name))
            continue
        slots = name2slot(p_name)
        if slots:
            update_slot(slots, p_array)
        param_num = np.prod(p_array.shape)
        num_sum = num_sum + param_num
        var_print('grad para', p_array, p_name, p_name, args.detail, logger)

    for p_name in param_name_list:
        p_array = np.array(train_exe.scope.find_var(p_name).get_tensor())
        slots = name2slot(p_name)
        if slots:
            update_slot(slots, p_array)
        param_num = np.prod(p_array.shape)
        num_sum = num_sum + param_num
        var_print('para', p_array, p_name, p_name, args.detail, logger)
    record_slot(logger)
    logger.info("total param num: {0}".format(num_sum))


def prepare_batch_input(batch, args):
    x = batch['token_ids']
    x_r = batch['token_ids_reverse']
    y = batch['next_token_id']
    y_r = batch['next_token_id_reverse']
    inst = []
    for i in range(len(x)):
        inst.append([x[i], y[i], x_r[i], y_r[i]])
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

    # Use test set as validation each pass
    total_loss = 0.0
    total_cnt = 0
    n_batch_cnt = 0
    n_batch_loss = 0.0
    val_feed_list = [
        inference_program.global_block().var(var_name)
        for var_name in feed_order
    ]
    val_feeder = fluid.DataFeeder(val_feed_list, place)
    dev_data_iter = lambda: dev_data.iter_batches(args.batch_size, args.num_steps)
    dev_reader = read_multiple(dev_data_iter, dev_count)

    for batch_id, batch_list in enumerate(dev_reader(), 1):
        feed_data = batch_reader(batch_list, args)
        val_fetch_outs = parallel_executor.run(
            feed=list(val_feeder.feed_parallel(feed_data, dev_count)),
            fetch_list=[loss.name],
            return_numpy=False)
        total_loss += np.array(val_fetch_outs[0]).sum()

        n_batch_cnt += len(np.array(val_fetch_outs[0]))
        total_cnt += len(np.array(val_fetch_outs[0]))
        n_batch_loss += np.array(val_fetch_outs[0]).sum()
        log_every_n_batch = args.log_interval
        if log_every_n_batch > 0 and batch_id % log_every_n_batch == 0:
            logger.info('Average dev loss from batch {} to {} is {}'.format(
                batch_id - log_every_n_batch + 1, batch_id, "%.10f" % (
                    n_batch_loss / n_batch_cnt)))
            n_batch_loss = 0.0
            n_batch_cnt = 0
        batch_offset = 0

    ppl = np.exp(total_loss / total_cnt)
    return ppl


def train():
    args = parse_args()
    if args.enable_ce:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    data_path = args.data_path
    logger.info("begin to load data")
    vocab = data.Vocabulary(args.vocab_path, validate_file=True)
    vocab_size = vocab.size
    train_data = data.BidirectionalLMDataset(
        args.train_path, vocab, test=False, shuffle_on_load=False)
    valid_data = data.BidirectionalLMDataset(
        args.test_path, vocab, test=True, shuffle_on_load=False)

    logger.info("finished load data")

    logger.info('Initialize the model...')

    if not args.use_gpu:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    # clone from default main program and use it as the validation program
    # build model
    main_program = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        main_program.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed
    with fluid.program_guard(main_program, startup_prog):
        with fluid.unique_name.guard():
            # Training process
            loss, feed_order, grad_vars, grad_vars_name = lm_model.lm_model(
                hidden_size,
                vocab_size,
                batch_size,
                num_layers=args.num_layers,
                num_steps=args.num_steps,
                init_scale=args.init_scale,
                args=args)
            inference_program = main_program.clone(for_test=True)
            '''
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=args.max_grad_norm))
            '''

            # build optimizer
            if args.optim == 'adagrad':
                optimizer = fluid.optimizer.Adagrad(
                    learning_rate=args.learning_rate, epsilon=1.0e-6)
            elif args.optim == 'sgd':
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
            exe_strategy = fluid.parallel_executor.ExecutionStrategy()
            if args.para_print:
                exe_strategy.num_threads = 1
                debug_init(main_program, grad_vars, grad_vars_name)
                with open("program.desc", 'w') as f:
                    print(str(framework.default_main_program()), file=f)
            parallel_executor = fluid.ParallelExecutor(
                main_program=main_program,
                use_cuda=bool(args.use_gpu),
                exec_strategy=exe_strategy)
            load_params(main_program, parallel_executor, place, logger, args)
            print_para(main_program, parallel_executor, logger, args)

            # get train epoch size
            log_interval = args.log_interval
            total_time = 0.0
            for epoch_id in range(args.max_epoch):
                start_time = time.time()
                logger.info("epoch id {}".format(epoch_id))
                train_data_iter = lambda: train_data.iter_batches(batch_size, args.num_steps)
                train_reader = read_multiple(train_data_iter, dev_count)

                total_loss = 0
                total_num = 0
                n_batch_loss = 0.0
                for batch_id, batch_list in enumerate(train_reader(), 1):
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
                        print_para(main_program, parallel_executor, logger,
                                   args)
                        ppl = np.exp(total_loss / total_num)
                        logger.info("ppl {} {} ".format(batch_id, ppl))
                    if batch_id > 0 and batch_id % args.dev_interval == 0:
                        valid_ppl = eval(valid_data, inference_program,
                                         feed_order, dev_count, loss, place,
                                         logger, args)
                        logger.info("valid ppl {}".format(valid_ppl))
                    if args.detail and batch_id > 10:
                        exit()

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
