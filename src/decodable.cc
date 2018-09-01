// Create at 2017-02-23

#include "decodable.h"

#include <math.h>
#include "am.h"

namespace pocketkaldi {

Decodable::Decodable(
    AcousticModel *am,
    float prob_scale,
    const MatrixBase<float> &feats): 
        am_(am),
        prob_scale_(prob_scale),
        feats_(&feats) {
}

Decodable::~Decodable() {
  am_ = nullptr;
  feats_ = nullptr;
  prob_scale_ = 0.0f;
}

float Decodable::LogLikelihood(int frame, int trans_id) {
  if (log_prob_.NumRows() == 0) {
    // We need to compute log_prob here
    Compute();
  }

  int pdf_id = am_->TransitionIdToPdfId(trans_id);
  SubVector<float> frame_prob = log_prob_.Row(frame);
  return frame_prob(pdf_id);
}

bool Decodable::IsLastFrame(int frame) {
  if (log_prob_.NumRows() == 0) {
    // We need to compute log_prob here
    Compute();
  }

  return frame >= log_prob_.NumRows() - 1;
}

void Decodable::Compute() {
  am_->Compute(*feats_, &log_prob_);
  log_prob_.Scale(prob_scale_);
}

}  // namespace pocketkaldi
