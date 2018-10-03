// Created at 2018-09-28

#ifndef POCKETKALDI_LM_FST_H_
#define POCKETKALDI_LM_FST_H_

#include "fst.h"

namespace pocketkaldi {

// Fst for language model, including deterministic on demand for back-off arcs,
class LmFst : public Fst {
 public:
  static constexpr char kLmFst[] = "pk::fst_lm";

  // Get out-going arc of state with ilabel. For LM, we will follow the back-off
  // arc automatically when there is no arc of ilabel in current state
  // If no matched arc even follow the back-off arc, return false. 
  bool GetArc(int state, int ilabel, Arc *arc) const override;

  // Get the final score of state. If current state is not a final state. It
  // will flollow the back-off arc automatically
  float Final(int state_id) const override;

 private:
  // Get the backoff arc for given state. If there is no back-off arc return
  // nullptr
  const Arc *GetBackoffArc(int state) const;
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_FST_H_
