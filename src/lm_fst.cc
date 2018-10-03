// Created at 2018-09-30

#include "lm_fst.h"

#include <algorithm>
#include <cmath>

namespace pocketkaldi {

const Fst::Arc *LmFst::GetBackoffArc(int state) const {
  int num_arcs = CountArcs(state);
  if (num_arcs == 0) return nullptr;

  int start_idx = state_idx_[state];
  const Arc *first = &arcs_[start_idx];

  if (first->input_label != 0) return nullptr;
  return first;
}

bool LmFst::GetArc(int state, int ilabel, Arc *arc) const {
  assert(ilabel != 0 && "invalid ilabel");

  if (Fst::GetArc(state, ilabel, arc)) {
    return true;
  } else {
    const Arc *backoff_arc = GetBackoffArc(state);
    if (backoff_arc) {
      if (!GetArc(backoff_arc->next_state, ilabel, arc)) return false;
      arc->weight += backoff_arc->weight;
      return true;
    } else {
      return false;
    }
  }
}

float LmFst::Final(int state_id) const {
  float final = Fst::Final(state_id);
  if (std::isfinite(final)) {
    return final;
  } else {
    // Follow the backoff arc
    const Arc *backoff_arc = GetBackoffArc(state_id);
    if (backoff_arc) {
      float final_w = Final(backoff_arc->next_state);
      if (!std::isfinite(final_w)) return INFINITY;
      return final_w + backoff_arc->weight;
    } else {
      return INFINITY;
    }
  }
}

}  // namespace pocketkaldi
