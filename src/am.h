// Created at 2017-03-22

#ifndef POCKETKALDI_AM_H_
#define POCKETKALDI_AM_H_

#include <deque>
#include <vector>
#include "nnet.h"
#include "util.h"
#include "configuration.h"

#define PK_AM_SECTION "AM~0"

using pocketkaldi::Nnet;

namespace pocketkaldi {

// Acoustic model in ASR, inculde
//   - Neural network model
//   - Prior for each CD-state
//   - Map from transition-id to pdf-id
class AcousticModel {
 public:
  // Stores the instance data of AM
  class Instance;

  // Indicates using all available frames as a batch
  static constexpr int kBatchSizeAll = -1;

  AcousticModel();
  ~AcousticModel();

  // Read AcousticModel from configuration file
  Status Read(const Configuration &conf);

  // Gets the map from transition-id to pdf-id
  const Vector<int32_t> &TransitionPdfIdMap() const {
    return tid2pdf_;
  }

  // Compute the log-likelihood of the feature matrix
  void Process(Instance *inst,
               const VectorBase<float> &frame_feat,
               Matrix<float> *log_prob) const;

  // Close the stream and compute the remained frames in buffer
  void EndOfStream(Instance *inst, Matrix<float> *log_prob) const;
 
  // Number of PDFs in this AM
  int num_pdfs() const { return num_pdfs_; }

 private:
  Nnet nnet_;
  Vector<float> log_prior_;
  int left_context_;
  int right_context_;
  int chunk_size_;
  int num_pdfs_;
  Vector<int32_t> tid2pdf_;


  // Add a frame of featue into the back of feats_buffer
  void AppendFrame(Instance *inst, const VectorBase<float> &frame_feat) const;

  // Returns true if a batch is available to compute
  bool BatchAvailable(Instance *inst) const;

  // Compute the log_prob of a batch, if batch_size == kBatchSizeAll, compute
  // log_prob of all available frames
  void ComputeBatch(Instance *inst,
                    int batch_size,
                    Matrix<float> *log_prob) const;
};

// Stores the instance data of AM
class AcousticModel::Instance {
 public:
  Instance();

 private:
  bool started;
  std::deque<Vector<float>> feats_buffer;

  friend class AcousticModel;
  DISALLOW_COPY_AND_ASSIGN(Instance);
};


}  // namespace pocketkaldi


#endif
