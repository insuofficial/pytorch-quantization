#!/bin/bash
DATA_DIR=/data/imagenet

model=resnet50
experiment=$model-quantize
batch_size=64
epochs=100
num_gpu=4

OMP_NUM_THREADS=8 \
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0\
  --nnodes=1 \
  --nproc_per_node=$num_gpu \
train.py \
  --model $model \
  --pretrained \
  --torchcompile \
  --data-dir $DATA_DIR \
  --num-classes 1000 \
  --batch-size $batch_size \
  --epochs $epochs \
  --opt sgd \
  --momentum 0.9 \
  --weight-decay 0 \
  --sched cosine \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --warmup-epochs 0 \
  --cooldown-epochs $epochs \
  --amp \
  --pin-mem \
  --workers 4 \
  --experiment $experiment \
  --log-interval 100 \
  --quantize \
  --calibration-step 10 \
  --weight-quantizer uniform \
  --weight-observer minmax \
  --weight-scheme per-tensor \
  --weight-dtype int8 \
  --act-quantizer uniform \
  --act-observer minmax \
  --act-scheme per-channel \
  --act-dtype uint8
