// nnetbin/nnet-forward-multistream.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits>

#include "nnet/nnet-lstm-projected.h"
#include "nnet/nnet-blstm-projected.h"
#include "nnet/nnet-blclstm-projected.h"
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through LSTM and BLCLSTM Recurrent Neural Network by multi-streams to parallel.\n"
        "\n"
        "Usage:  nnet-forward-multistream [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward-multistream nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    int32 batch_size = 20;
    po.Register("batch-size", &batch_size, "low latency Bidirectional LSTM batch size"); 

    int num_stream=8;
    po.Register("num-stream", &num_stream, "if set the number of streams to parallel like LSTM RNN"); 
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    //optionally remove softmax
    if (no_softmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() ==
        kaldi::nnet1::Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveComponent(nnet.NumComponents()-1);
    }
    //check for some non-sense option combinations
    if (apply_log && no_softmax) {
      KALDI_ERR << "Nonsense option combination : --apply-log=true and --no-softmax=true";
    }
    if (apply_log && nnet.GetComponent(nnet.NumComponents()-1).GetType() !=
        kaldi::nnet1::Component::kSoftmax) {
      KALDI_ERR << "Used --apply-log=true, but nnet " << model_filename 
                << " does not have <softmax> as last component!";
    }
    
    PdfPrior pdf_prior(prior_opts);
    if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
      KALDI_ERR << "Option --class-frame-counts has to be used together with "
                << "--no-softmax or --apply-log";
    }

    // disable dropout
    nnet_transf.SetDropoutRate(0.0);
    nnet.SetDropoutRate(0.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    //  book-keeping for multi-streams
    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<int> curt(num_stream, 0);
    std::vector<int> lent(num_stream, 0);
    std::vector<int> new_utt_flags(num_stream, 0);
    int32 feat_dim = nnet.InputDim();
    Matrix<BaseFloat> feat;

    CuMatrix<BaseFloat> feat_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;
    Matrix<BaseFloat> nnet_out_host_sub;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;

    int32 max_len = 0;
    int32 cur_stream = num_stream;

    int32 cur_batch_size = batch_size;

    while (cur_stream != 0 && cur_stream == num_stream) {
        // loop over all streams, check if any stream reaches the end of its utterance,
        // if any, feed the exhausted stream with a new utterance, update book-keeping infos
        max_len = 0;
        cur_stream = 0;
        cur_batch_size = batch_size;

        for (int s = 0; s < num_stream; s++) {
            // else, this stream exhausted, need new utterance
            if (!feature_reader.Done()) {
                keys[s]  = feature_reader.Key();
                feats[s] = feature_reader.Value();
                curt[s] = 0;
                lent[s] = feats[s].NumRows();
                feature_reader.Next();
                ++cur_stream;
                if (max_len < lent[s]) {
                    max_len = lent[s];
                }
                ;
            } else {
                break;
            }
        }
        if (cur_stream == 0) {
            continue;
        } else if (cur_stream < num_stream) {
           for (int s = cur_stream; s < num_stream; ++s) {
               keys[s] = keys[cur_stream - 1];
               feats[s] = feats[cur_stream - 1];
               curt[s] = 0;
               lent[s] = lent[cur_stream - 1];
           }
        }

        nnet.SetSeqLengths(lent);

        for (int s = 0; s < num_stream; ++s) {
            new_utt_flags[s] = 1;
        }
        nnet.ResetStreams(new_utt_flags);

        nnet_out.Resize(max_len * num_stream, nnet.OutputDim());         
        // online decoding

        if (cur_batch_size == 0) {
            cur_batch_size = max_len;
        }
 
        CuMatrix<BaseFloat> nnet_out_batch;
        int nframes = 0;
        feat.Resize(cur_batch_size * num_stream, feat_dim);
        while ((nframes + cur_batch_size) <= max_len) {
            // fill a multi-stream bptt batch
            for (int t = 0; t < cur_batch_size; t++) {
                for (int s = 0; s < num_stream; s++) {
                    // feat shifting & padding
                    if (curt[s] < lent[s]) {
                        feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]));
                    } else {
                        feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s]-1));
                    }
                    curt[s]++;
                }
            }

            // apply optional feature transform
            nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feat_transf);

            // forward pass
            nnet.Feedforward(feat_transf, &nnet_out_batch);

            // copy to output
            nnet_out.RowRange(nframes*num_stream, cur_batch_size * num_stream).CopyFromMat(nnet_out_batch);

            nframes += cur_batch_size;
        }

        KALDI_LOG << "nframes = " << nframes <<",batch_size = "<< cur_batch_size  << ",max_len = " << max_len;
 
        if ((nframes < max_len) && ((nframes + cur_batch_size) > max_len)) {
            int remainframes = max_len - nframes;
            feat.Resize(remainframes * num_stream, feat_dim);
            // fill a multi-stream bptt batch
            for (int t = 0; t < remainframes; t++) {
                for (int s = 0; s < num_stream; s++) {
                    // feat shifting & padding
                    if (curt[s] < lent[s]) {
                        feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]));
                    } else {
                        feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s]-1));
                    }
                    curt[s]++;
                }
            }
            // apply optional feature transform
            nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feat_transf);

            // forward pass
            nnet.Feedforward(feat_transf, &nnet_out_batch);
          
            // copy to output
            nnet_out.RowRange(nframes*num_stream, remainframes*num_stream).CopyFromMat(nnet_out_batch);
        }
        // convert posteriors to log-posteriors
        if (apply_log) {
            nnet_out.ApplyLog();
        }

        // subtract log-priors from log-posteriors to get quasi-likelihoods
        if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
            pdf_prior.SubtractOnLogpost(&nnet_out);
        }

        //download from GPU
        nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
        nnet_out.CopyToMat(&nnet_out_host);

        //KALDI_LOG << "size of NNET_OUT " << nnet_out_host.NumRows() <<", " << nnet_out_host.NumCols();
        //KALDI_LOG << nnet_out_host;
        for (int s = 0; s < cur_stream; s++) {
            nnet_out_host_sub.Resize(lent[s],nnet_out_host.NumCols());
            for (int t = 0; t < lent[s]; ++t) {
                nnet_out_host_sub.Row(t).CopyFromVec(nnet_out_host.Row(t * num_stream + s));
            }
#ifdef DEBUG
            //check for NaN/inf
            for (int32 r = 0; r < nnet_out_host_sub.NumRows(); r++) {
              for (int32 c = 0; c < nnet_out_host_sub.NumCols(); c++) {
                BaseFloat val = nnet_out_host_sub(r,c);
                if (val != val) 
                  KALDI_ERR << "NaN in NNet output of : " << keys(s);
                if (val == std::numeric_limits<BaseFloat>::infinity())
                  KALDI_ERR << "inf in NNet coutput of : " << keys(s);
              }
            }
#endif
            // write
            //KALDI_LOG << keys[s] << "," << lent[s] << "," << nnet_out_host_sub.NumRows() << "," << nnet_out_host_sub.NumCols();
            //KALDI_LOG << nnet_out_host_sub;
            feature_writer.Write(keys[s], nnet_out_host_sub);
            tot_t += lent[s];
        }

        // progress log
        num_done+=cur_stream;
        if (num_done % 100 == 0) {
            time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
        }
        //tot_t += mat.NumRows();
        //if (feature_reader.Done()) {
        //   break;
        //}
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
