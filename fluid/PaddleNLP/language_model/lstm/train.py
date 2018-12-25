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
name_dict['fw_layer2_gate_w'] = 32
name_dict['lstmp_1.w_0'] = 32
name_dict['lstmp_1.w_1'] = 33

name_dict['lstmp_2.b_0'] = 41
name_dict['bw_layer1_gate_w'] = 42
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
            [str(round(x, 10)) for x in [slot] + slot_dict[slot]]))


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
                    card = 0
                    #for scope in [train_exe.scope]:#train_exe.executor.local_scopes():
                    for scope in train_exe.executor.local_scopes():
                        tensor = scope.find_var(name).get_tensor()
                        shape = tensor.shape()
                        tensor_len = np.prod(shape)
                        new_array = p_array[offset:offset + tensor_len]
                        new_array = new_array.reshape(shape)
                        placex = fluid.CUDAPlace(card)
                        tensor.set(new_array.astype(np.float32), placex)
                        logger.info('card {} loaded {}[{}] from {}[{}:{}]'.
                                    format(card, name, shape, data, offset,
                                           offset + tensor_len))
                        card = card + 1
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


def debug_print(scope, logger, args):
    if not args.para_print:
        return
    for name_pair in names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not scope.find_var(name):
            logger.info("var: {0} not find".format(p_name))
            continue
        p_array = np.array(scope.find_var(name).get_tensor()).astype('float64')
        var_print('var', p_array, p_name, name, args.detail, logger)
    for name_pair in grad_names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not scope.find_var(name):
            logger.info("grad: {0} not find".format(p_name))
            continue
        p_array = np.array(scope.find_var(name).get_tensor()).astype('float64')
        var_print('grad', p_array, p_name, name, args.detail, logger)


def vars_print(logger, args, vars=None, grad_vars=None):
    if not args.para_print:
        return
    for var, vname in zip(vars):
        name, p_name = vname
        p_array = np.array(var).astype('float64')
        var_print('var', p_array, p_name, name, args.detail, logger)
    for grad, gname in zip(grad_vars):
        name, p_name = gname
        p_array = np.array(grad).astype('float64')
        var_print('grad', p_array, p_name, name, args.detail, logger)


def print_para(train_prog, train_exe, logger, optimizer=None, args=None):
    if not args.para_print:
        return
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    card = 0
    for scope in train_exe.executor.local_scopes():
        init_slot()
        num_sum = 0
        logger.info('card {}'.format(card))
        debug_print(scope, logger, args)
        for p_name in param_name_list:
            p_name = p_name + '@GRAD'
            if not scope.find_var(p_name):
                logger.info("grad para: {0} not find".format(p_name))
                #import pdb; pdb.set_trace()
                continue
            try:
                p_array = np.array(scope.find_var(p_name).get_tensor())
            except:
                #import pdb; pdb.set_trace()
                logger.info("grad para: {0} failed".format(p_name))
                continue
            param_num = np.prod(p_array.shape)
            var_print('grad para', p_array, p_name, p_name, args.detail, logger)
        if optimizer:
            for p_name in param_name_list:
                acc_str = 'moment'
                acc = optimizer._accumulators[acc_str][p_name]
                p_array = np.array(scope.find_var(acc.name).get_tensor())
                var_print(acc_str, p_array, p_name, acc.name, args.detail,
                          logger)
        for p_name in param_name_list:
            p_array = np.array(scope.find_var(p_name).get_tensor())
            slots = name2slot(p_name)
            if slots:
                update_slot(slots, p_array)
            param_num = np.prod(p_array.shape)
            num_sum = num_sum + param_num
            var_print('para', p_array, p_name, p_name, args.detail, logger)
        record_slot(logger)
        logger.info("total param num: {0}".format(num_sum))

        card = card + 1


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
        #res.append(prepare_batch_input(batch_list[0], args))
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


def eval(vocab, inference_program, feed_order, dev_count, loss, place, logger,
         args):
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
    dev_data = data.BidirectionalLMDataset(
        args.test_path, vocab, test=True, shuffle_on_load=False)
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
    logger.info("begin to load vocab")
    vocab = data.Vocabulary(args.vocab_path, validate_file=True)
    vocab_size = vocab.size
    logger.info("finished load vocab")

    logger.info('build the model...')
    # clone from default main program and use it as the validation program
    # build model
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        train_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # Training process
            model = lm_model.LanguageModel(args, vocab_size)
            model.build()
            infer_prog = train_prog.clone(for_test=True)
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=args.max_grad_norm))

            # build optimizer
            if args.optim == 'adagrad':
                optimizer = fluid.optimizer.Adagrad(
                    learning_rate=args.learning_rate,
                    epsilon=0.0,
                    initial_accumulator_value=1.0)
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
            optimizer.minimize(model.loss * args.num_steps)

            # initialize parameters
            place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
            exe = Executor(place)

    if args.local:
        logger.info("local start_up:")
        train_loop(args, logger, vocab, train_prog, startup_prog, infer_prog, model, optimizer)
    else:
        if args.update_method == "nccl2":
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            if args.test_nccl:
                worker_endpoints_env = os.getenv("PADDLE_WORK_ENDPOINTS")
                worker_endpoints = worker_endpoints_env.split(',')
                trainers_num=len(worker_endpoints)
                current_endpoint=worker_endpoints[trainer_id]
            else:
		port = os.getenv("PADDLE_PORT")
		worker_ips = os.getenv("PADDLE_TRAINERS")
		worker_endpoints = []
		for ip in worker_ips.split(","):
		    worker_endpoints.append(':'.join([ip, port]))
                worker_endpoints_env = ','.join(worker_endpoints)
		trainers_num = len(worker_endpoints)
		current_endpoint = os.getenv("POD_IP") + ":" + port
            if trainer_id == 0:
                logger.info("train_id == 0, sleep 60s")
                time.sleep(60)


            logger.info("trainers_num:{}".format(trainers_num))
            logger.info("worker_endpoints:{}".format(worker_endpoints))
            logger.info("current_endpoint:{}".format(current_endpoint))
            config = fluid.DistributeTranspilerConfig()
            config.mode = "nccl2"
            t = fluid.DistributeTranspiler(config=config)
            t.transpile(
                trainer_id,
                trainers=worker_endpoints_env,
                current_endpoint=current_endpoint,
                program=train_prog,
                startup_program=startup_prog)
            train_loop(args, logger, vocab, train_prog, startup_prog, infer_prog, model, optimizer, trainers_num,
                       trainer_id, worker_endpoints)
        else:
            port = os.getenv("PADDLE_PORT", "6174")
            pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
            eplist = []
            for ip in pserver_ips.split(","):
                eplist.append(':'.join([ip, port]))
            pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
            trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
            current_endpoint = os.getenv("POD_IP") + ":" + port
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))

            logger.info("pserver_endpoints:{}".format(pserver_endpoints))
            logger.info("current_endpoint:{}".format(current_endpoint))
            logger.info("trainer_id:{}".format(trainer_id))
            logger.info("pserver_ips:{}".format(pserver_ips))
            logger.info("port:{}".format(port))

            t = fluid.DistributeTranspiler()
            t.transpile(
                trainer_id,
                pservers=pserver_endpoints,
                trainers=trainers,
                program=train_prog,
                startup_program=startup_prog)

            if training_role == "PSERVER":
                logger.info("distributed: pserver started")
                current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                    "PADDLE_PORT")
                if not current_endpoint:
                    logger.critical("need env SERVER_ENDPOINT")
                    exit(1)
                pserver_prog = t.get_pserver_program(current_endpoint)
                pserver_startup = t.get_startup_program(current_endpoint,
                                                        pserver_prog)

                exe.run(pserver_startup)
                exe.run(pserver_prog)
            elif training_role == "TRAINER":
                logger.info("distributed: trainer started")
                trainer_prog = t.get_trainer_program()
                train_loop(args, logger, vocab, train_prog, startup_prog, infer_prog, model, optimizer)
            else:
                logger.critical(
                    "environment var TRAINER_ROLE should be TRAINER os PSERVER")
                exit(1)


def train_loop(args,
               logger,
               vocab,
               main_prog,
               startup_prog,
               infer_prog,
               model,
               optimizer,
               nccl2_num_trainers=1,
               nccl2_trainer_id=0,
               worker_endpoints=None):
    # prepare device
    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    if not args.use_gpu:
        place = fluid.CPUPlace()
        import multiprocessing
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    if args.load_dir:
        logger.info('load from {}'.format(args.load_dir))
        fluid.io.load_persistables(
            exe, args.load_dir, main_program=main_prog)
    else:
        exe.run(startup_prog)

    # prepare data
    feed_list = [
        main_prog.global_block().var(var_name)
        for var_name in model.feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    logger.info('Training the model...')
    exe_strategy = fluid.parallel_executor.ExecutionStrategy()
    if args.para_print:
        exe_strategy.num_threads = 1
        debug_init(main_prog, model.grad_vars, model.grad_vars_name)
        with open("program.desc", 'w') as f:
            print(str(main_prog), file=f)
    parallel_executor = fluid.ParallelExecutor(
        loss_name=model.loss.name,
        main_program=main_prog,
        use_cuda=bool(args.use_gpu),
        exec_strategy=exe_strategy,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)
    load_params(main_prog, parallel_executor, place, logger, args)
    print_para(main_prog, parallel_executor, logger, optimizer, args)

    logger.info("begin to load data")
    train_data = data.BidirectionalLMDataset(
        args.train_path, vocab, test=(not args.shuffle), shuffle_on_load=False)
    logger.info("finished load vocab")

    # get train epoch size
    log_interval = args.log_interval
    total_time = 0.0
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    for epoch_id in range(args.max_epoch):
        start_time = time.time()
        logger.info("epoch id {}".format(epoch_id))
        train_data_iter = lambda: train_data.iter_batches(batch_size, args.num_steps)
        train_reader = read_multiple(train_data_iter, dev_count)

        total_loss = 0
        total_num = 0
        n_batch_loss = 0.0
        n_batch_cnt = 0
        last_hidden_values = np.zeros(
            (dev_count,
             args.num_layers * 2 * batch_size * args.embed_size),
            dtype='float32')
        last_cell_values = np.zeros(
            (dev_count, args.num_layers * 2 * batch_size * hidden_size),
            dtype='float32')
        for batch_id, batch_list in enumerate(train_reader(), 1):
            feed_data = batch_reader(batch_list, args)
            feed = list(feeder.feed_parallel(feed_data, dev_count))
            for i in range(dev_count):
                init_hidden_tensor = fluid.core.LoDTensor()
                if args.use_gpu:
                    placex = fluid.CUDAPlace(i)
                else:
                    placex = fluid.CPUPlace()
                init_hidden_tensor.set(last_hidden_values[i], placex)
                init_cell_tensor = fluid.core.LoDTensor()
                init_cell_tensor.set(last_cell_values[i], placex)

                feed[i]['init_hiddens'] = init_hidden_tensor
                feed[i]['init_cells'] = init_cell_tensor

            fetch_outs = parallel_executor.run(
                feed=feed,
                fetch_list=[
                    model.loss.name, model.last_hidden.name, model.last_cell.name
                ],  # + [x[0] for x in names] + [x[0] for x in grad_names],
                return_numpy=False)
            cost_train = np.array(fetch_outs[0]).mean()
            last_hidden_values = np.array(fetch_outs[1])
            last_hidden_values = last_hidden_values.reshape(
                (dev_count,
                 args.num_layers * 2 * batch_size * args.embed_size))
            last_cell_values = np.array(fetch_outs[2])
            last_cell_values = last_cell_values.reshape(
                (dev_count,
                 args.num_layers * 2 * batch_size * args.hidden_size))

            #vars = fetch_outs[2:2+len(names)]
            #grad_vars = fetch_outs[2+len(names):]

            total_num += args.batch_size * dev_count
            n_batch_loss += np.array(fetch_outs[0]).sum()
            #logger.info("n_batch_loss from {} to {} is {}, {} ".format(
            #    batch_id - log_interval, batch_id, n_batch_loss,
            #    np.array(fetch_outs[0]).sum()))
            n_batch_cnt += len(np.array(fetch_outs[0]))
            total_loss += cost_train * args.batch_size * dev_count

            if batch_id > 0 and batch_id % log_interval == 0:
                #vars_print(logger, args, vars=(vars, names), grad_vars=(grad_vars, grad_names))
                print_para(main_prog, parallel_executor, logger,
                           optimizer, args)
                ppl = np.exp(n_batch_loss / n_batch_cnt)
                logger.info("ppl from {} to {} is {} ".format(
                    batch_id - log_interval, batch_id, ppl))
                n_batch_loss = 0.0
                n_batch_cnt = 0
            if batch_id > 0 and batch_id % args.dev_interval == 0:
                valid_ppl = eval(vocab, infer_prog, model.feed_order,
                                 dev_count, model.loss, place, logger, args)
                logger.info("valid ppl {}".format(valid_ppl))
            if batch_id > 0 and batch_id % args.save_interval == 0:
                model_path = os.path.join("model_new/",
                                          str(batch_id + epoch_id))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                fluid.io.save_persistables(
                    executor=exe,
                    dirname=model_path,
                    main_program=main_prog)
            if args.detail and batch_id > 100:
                exit()

        end_time = time.time()
        total_time += end_time - start_time
        logger.info("train ppl {}".format(ppl))

        if epoch_id == args.max_epoch - 1 and args.enable_ce:
            logger.info("lstm_language_model_duration\t%s" %
                        (total_time / args.max_epoch))
            logger.info("lstm_language_model_loss\t%s" % ppl[0])

        model_path = os.path.join("model_new/", str(epoch_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(
            executor=exe, dirname=model_path, main_program=main_prog)
        valid_ppl = eval(vocab, infer_prog, model.feed_order,
                         dev_count, model.loss, place, logger, args)
        logger.info("valid ppl {}".format(valid_ppl))
    test_ppl = eval(vocab, infer_prog, model.feed_order, dev_count,
                    model.loss, place, logger, args)
    logger.info("test ppl {}".format(test_ppl))


if __name__ == '__main__':
    train()
