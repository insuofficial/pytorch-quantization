#!/bin/bash
DATA_DIR=/data/imagenet

model=resnet50
batch_size=256
num_gpu=2

OMP_NUM_THREADS=8 \
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0\
  --nnodes=1 \
  --nproc_per_node=$num_gpu \
validate.py \
  --model $model \
  --pretrained \
  --data-dir $DATA_DIR \
  --num-classes 1000 \
  --batch-size $batch_size \
  --amp \
  --pin-mem \
  --workers 8 \
  --log-freq 10 \
  --quantize \
  --calibration-step 10 \
  --weight-quantizer uniform \
  --weight-observer minmax \
  --weight-scheme per-tensor \
  --weight-dtype int8 \
  --act-quantizer uniform \
  --act-observer minmax \
  --act-scheme per-channel \
  --act-dtype int8

