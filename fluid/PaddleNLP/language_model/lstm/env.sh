#!/bin/bash 
set -xe

#library
export PATH=$PWD/python/bin/:$PATH
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/nccl_2.3.5/lib/:$LD_LIBRARY_PATH

#http_proxy
unset http_proxy
unset https_proxy

#paddle envs
export PADDLE_PSERVER_PORT=9184
export PADDLE_TRAINER_IPS=${iplist} 
export PADDLE_CURRENT_IP=`hostname -i`

iparray=(${iplist//,/ })
for i in "${!iparray[@]}"; do
    if [ ${iparray[$i]} == ${PADDLE_CURRENT_IP} ]; then
        export PADDLE_TRAINER_ID=$i
    fi
done

export TRAINING_ROLE=TRAINER
#export PADDLE_PSERVERS=127.0.0.1
export PADDLE_INIT_TRAINER_COUNT=${#iparray[@]}
export PADDLE_PORT=${PADDLE_PSERVER_PORT}
export PADDLE_TRAINERS=${PADDLE_TRAINER_IPS}
export POD_IP=${PADDLE_CURRENT_IP}
export PADDLE_TRAINERS_NUM=${PADDLE_INIT_TRAINER_COUNT}

#paddle debug envs
#export GLOG_v=7
#export GLOG_logtostderr=1

#nccl debug envs
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_IB_GDR_LEVEL=4
export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth2


