#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

sort_by_len=true

#dev=data-fbankpitch/test
#train=data-fbankpitch/train

dev=data-fmllr-tri3/test
train=data-fmllr-tri3/train

dev_original=data/test
train_original=data/train

gmm=exp/tri3

stage=0
. utils/parse_options.sh || exit 1;

false &&
{
# Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi
}

dir=exp/blstm4f
mkdir -p $dir

#false &&
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/blstm4f
  ali=${gmm}_ali

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --network-type blstm --learn-rate 0.00004 --momentum 0.9 \
      --feat-type plain --splice 0 \
      --scheduler-opts "--halving-factor 0.5" \
      --train-tool "nnet-train-multistream-perutt" \
      --train-tool-opts "--num-streams=10 --max-frames=15000" \
      --proto-opts "--cell-dim 320 --proj-dim 200 --num-layers 2" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode_test || exit 1;
fi
}

#dir=exp/blstm4f
#steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
#    $gmm/graph $dev $dir/decode_test || exit 1;

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
