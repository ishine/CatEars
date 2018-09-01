// Create at 2017-02-23

#ifndef POCKETKALDI_DECODABLE_H_
#define POCKETKALDI_DECODABLE_H_

#include "decodable.h"

#include <assert.h>
#include "am.h"
#include "matrix.h"

namespace pocketkaldi {

// Provide the log-likelihood for each frame and transition-id
class Decodable {
 public:
  Decodable(
      AcousticModel *am,
      float prob_scale,
      const MatrixBase<float> &feats);

  ~Decodable();

  // Log-likelihood for each frame and trans_id
  float LogLikelihood(int frame, int trans_id);

  // Returns true if it is the last frame
  bool IsLastFrame(int frame);

  // Compute the log_prob for each frame
  void Compute();
 private:
  AcousticModel *am_;
  const MatrixBase<float> *feats_;
  Matrix<float> log_prob_;
  float prob_scale_;

};

}  // pocketkaldi

#endif  // POCKETKALDI_DECODABLE_H_
