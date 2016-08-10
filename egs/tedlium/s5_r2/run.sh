#!/bin/bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014 Nickolay V. Shmyrev
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0
#

# TODO : use pruned trigram?

. cmd.sh
. path.sh

#nj=40
nj=20
decode_nj=8

stage=0
. utils/parse_options.sh # accept options


# Run the DNN recipe on fMLLR feats:
local/nnet/run_blstm.sh || exit 1
# for decode_dir in "exp/dnn4_pretrain-dbn_dnn/decode_test" "exp/dnn4_pretrain-dbn_dnn_smbr_i1lats/decode_test_it4"; do
#  steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/test $decode_dir $decode_dir.rescore
# done
# DNN recipe with bottle-neck features
# local/nnet/run_dnn_bn.sh
# Rescore with 4-gram LM:
# decode_dir=exp/dnn8f_BN_pretrain-dbn_dnn_smbr/decode_test_it4
# steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/test $decode_dir $decode_dir.rescore || exit 1

# Nnet2 multisplice recipe
# local/online/run_nnet2_ms_perturbed.sh || exit 1;
# Run discriminative training on the top of multisplice recipe
# local/online/run_nnet2_ms_disc.sh || exit 1;

# Nnet3 TDNN recipe
# local/nnet3/run_tdnn.sh
# local/nnet3/run_tdnn_discriminative.sh

#local/chain/run_tdnn.sh


echo success...
exit 0
