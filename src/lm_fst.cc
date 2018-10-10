// Created at 2018-09-30

#include "lm_fst.h"

#include <algorithm>
#include <cmath>
#include "symbol_table.h"

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

DeltaLmFst::DeltaLmFst(
    const Vector<float> *small_lm,
    const LmFst *lm,
    const SymbolTable *symbol_table):
        small_lm_(small_lm),
        lm_(lm) {
  bos_symbol_ = symbol_table->bos_id();
  eos_symbol_ = symbol_table->eos_id();
}

int DeltaLmFst::StartState() const {
  int start_state = lm_->start_state();
  Fst::Arc arc;
  if (lm_->GetArc(start_state, bos_symbol_, &arc)) {
    return arc.next_state;
  } else {
    PK_WARN("lm_ didn't have symbol <s> as input");
    return start_state;
  }
}

bool DeltaLmFst::GetArc(int state, int ilabel, Fst::Arc *arc) const {
  if (lm_->GetArc(state, ilabel, arc)) {
    arc->weight -= (*small_lm_)(ilabel);
    return true;
  } else {
    return false;
  }
}

float DeltaLmFst::Final(int state_id) const {
  Fst::Arc arc;
  if (lm_->GetArc(state_id, eos_symbol_, &arc)) {
    return lm_->Final(arc.next_state) + arc.weight - (*small_lm_)(eos_symbol_);
  } else {
    return INFINITY;
  }
}

}  // namespace pocketkaldi
