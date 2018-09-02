// Created at 2017-03-22

#ifndef POCKETKALDI_AM_H_
#define POCKETKALDI_AM_H_

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
  AcousticModel();
  ~AcousticModel();

  // Read AcousticModel from configuration file
  Status Read(const Configuration &conf);

  // Convert transition-id to pdf-id
  int TransitionIdToPdfId(int transition_id) const {
    return tid2pdf_(transition_id);
  }

  // Compute the log-likelihood of the feature matrix
  void Compute(const MatrixBase<float> &frames, Matrix<float> *log_prob);
 
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

  // Prepare the input chunk from begin_frame. Chunk size is min(chunk_size_,
  // frames.NumRow() - begin_frame). And return the chunk size;
  int PrepareChunk(
      const MatrixBase<float> &frames,
      int begin_frame,
      Matrix<float> *chunk);
};

}  // namespace pocketkaldi


#endif
