#!/bin/bash
set -xe

export PADDLE_WORK_ENDPOINTS=127.0.0.1:9184,127.0.0.1:9185

#for ubuntu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/workspace/brpc:/usr/lib/x86_64-linux-gnu/
export PYTHONPATH=$PYTHONPATH:/paddle/build/build_bertgather2_RelWithDebInfo_gpu/python
#echo $LD_LIBRARY_PATH
export FLAGS_nccl_dir=/usr/lib/x86_64-linux-gnu/

# for cenos6u3
#export PYTHONPATH=$PYTHONPATH:/paddle/build/build_bertgather_centos6u3_release_gpu/python
#export PATH=$PWD/python/bin/:$PATH
#export LD_LIBRARY_PATH=$PWD/nccl_2.3.5/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v6/cuda/lib64:$LD_LIBRARY_PATH

unset http_proxy https_proxy

#paddle debug envs
export GLOG_v=0
export GLOG_logtostderr=0

#nccl debug envs
export NCCL_DEBUG=INFO
#export NCCL_DEBUG=VERSION
export NCCL_IB_DISABLE=1
#export NCCL_IB_GDR_LEVEL=4
export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth2
export NCCL_P2P_DISABLE=1


export PADDLE_TRAINER_ID=0
export CUDA_VISIBLE_DEVICES=0,1,2
./test_model_nccl.sh

export PADDLE_TRAINER_ID=1
export CUDA_VISIBLE_DEVICES=4,5,6
./test_model_nccl.sh
