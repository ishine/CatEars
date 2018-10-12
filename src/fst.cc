// Created at 2016-11-24

#include "fst.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <array>
#include <algorithm>
#include "symbol_table.h"

namespace pocketkaldi {

const char *Fst::kSectionName = "pk::fst_0";

FstArc::FstArc() {}
FstArc::FstArc(int next_state, int ilabel, int olabel, float weight): 
    next_state(next_state),
    input_label(ilabel),
    output_label(olabel),
    weight(weight) {}


Fst::Fst(): start_state_(0) {}
Fst::~Fst() {}

Fst::ArcIterator::ArcIterator(int base, int total, const FstArc *arcs) :
    base_(base),
    cnt_pos_(0),
    total_(total),
    arcs_(arcs) {
}
Fst::ArcIterator::~ArcIterator() {
  arcs_ = nullptr;
  base_ = 0;
  cnt_pos_ = 0;
  total_ = 0;
}

int Fst::StartState() const {
  return start_state_;
}

float Fst::Final(int state_id) const {
  assert(state_id < final_.size());
  return final_[state_id];
}

Status Fst::Read(util::ReadableFile *fd) {
  Status status;

  // Checks section name
  int section_size;
  std::array<char, 32> section_name;
  status = fd->Read(section_name.data(), 32);
  if (!status.ok()) return status;
  section_name.back() = '\0';
  if (std::string(section_name.data()) != kSectionName) {
    return Status::Corruption(fd->filename());
  }
  status = fd->ReadValue<int32_t>(&section_size);
  if (!status.ok()) return status;

  // Metadata
  int32_t state_number,
          arc_number,
          start_state;
  status = fd->ReadValue<int32_t>(&state_number);
  if (!status.ok()) return status;
  status = fd->ReadValue<int32_t>(&arc_number);
  if (!status.ok()) return status;  
  status = fd->ReadValue<int32_t>(&start_state);
  if (!status.ok()) return status;  
  start_state_ = start_state;

  // Check section size
  int expected_section_size =
      sizeof(state_number) +
      sizeof(arc_number) +
      sizeof(start_state) +
      state_number * (sizeof(final_.front()) + sizeof(state_idx_.front())) +
      arc_number * sizeof(FstArc);
  if (expected_section_size != section_size) {
    return Status::Corruption(util::Format(
        "section_size == {} expected, but {} found",
        expected_section_size,
        section_size));
  }

  // Final weight
  final_.resize(state_number);
  status = fd->Read(final_.data(), sizeof(final_.front()) * final_.size());
  if (!status.ok()) return status; 

  // State idx
  state_idx_.resize(state_number);
  status = fd->Read(
      state_idx_.data(),
      sizeof(state_idx_.front()) * state_idx_.size());
  if (!status.ok()) return status;

  // Arcs
  arcs_.resize(arc_number);
  status = fd->Read(arcs_.data(), sizeof(arcs_.front()) * arcs_.size());
  if (!status.ok()) return status;

  // Success
  return status;
}

int Fst::CountArcs(int state) const {
  int state_idx = state_idx_[state];
  if (state_idx < 0) return 0;

  int count;
  int next_state = -1;

  // Find the next state that have outcoming arcs
  for (int chk_state = state + 1; chk_state < state_idx_.size(); ++chk_state) {
    if (state_idx_[chk_state] > 0) {
      next_state = chk_state;
      break;
    }
  }
  int next_idx = next_state >= 0 ? state_idx_[next_state] : arcs_.size();
  return next_idx - state_idx;
}

bool Fst::GetArc(int state, int ilabel, FstArc *arc) const {
  int num_arcs = CountArcs(state);
  if (num_arcs == 0) return false;

  int start_idx = state_idx_[state];
  const FstArc *first = &arcs_[start_idx];
  const FstArc *last = &arcs_[start_idx + num_arcs];


  const FstArc *found_arc = std::lower_bound(
      first,
      last,
      FstArc(0, ilabel, 0, 0.0f), 
      [] (const FstArc &l, const FstArc &r) {
        return l.input_label < r.input_label;
      });
  if (found_arc == last || found_arc->input_label != ilabel) {
    return false;
  } else {
    // We have found this arc
    *arc = *found_arc;
    return true;
  }
}

Fst::ArcIterator Fst::IterateArcs(int state) const {
  assert(state < state_idx_.size() && state >= 0);
  int total_arcs = CountArcs(state);
  return ArcIterator(
    state_idx_[state],
    total_arcs,
    arcs_.data());
}

const FstArc *Fst::ArcIterator::Next() {
  if (cnt_pos_ < total_) {
    const FstArc *arc = &arcs_[base_ + cnt_pos_];
    ++cnt_pos_;
    return arc;
  } else {
    return nullptr;
  }
}

void LmFst::InitBucket0() {
  // Optimize for special state 0
  ArcIterator arc_iter = IterateArcs(0);
  const FstArc *arc = nullptr;
  int max_ilabel = 0;
  while ((arc = arc_iter.Next()) != nullptr) {
    if (arc->input_label > max_ilabel) {
      max_ilabel = arc->input_label;
    }
  }

  // Initalize bucket_0_
  bucket_0_.resize(max_ilabel + 1);
  for (FstArc &arc : bucket_0_) {
    arc.input_label = -1;
  }

  // Fill bucket_0_
  arc_iter = IterateArcs(0);
  while ((arc = arc_iter.Next()) != nullptr) {
    bucket_0_[arc->input_label] = *arc;
  }
}

const FstArc *LmFst::GetBackoffArc(int state) const {
  int num_arcs = CountArcs(state);
  if (num_arcs == 0) return nullptr;

  int start_idx = state_idx_[state];
  const FstArc *first = &arcs_[start_idx];

  if (first->input_label != 0) return nullptr;
  return first;
}

bool LmFst::GetArc(int state, int ilabel, FstArc *arc) const {
  assert(ilabel != 0 && "invalid ilabel");

  // We have special optimize for state 0
  if (state == 0 && ilabel < bucket_0_.size()) {
    if (bucket_0_[ilabel].input_label == ilabel) {
      *arc = bucket_0_[ilabel];
      return true;
    }
  }

  if (Fst::GetArc(state, ilabel, arc)) {
    return true;
  } else {
    const FstArc *backoff_arc = GetBackoffArc(state);
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
    const FstArc *backoff_arc = GetBackoffArc(state_id);
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
  int start_state = lm_->StartState();
  FstArc arc;
  if (lm_->GetArc(start_state, bos_symbol_, &arc)) {
    return arc.next_state;
  } else {
    PK_WARN("lm_ didn't have symbol <s> as input");
    return start_state;
  }
}

bool DeltaLmFst::GetArc(int state, int ilabel, FstArc *arc) const {
  if (lm_->GetArc(state, ilabel, arc)) {
    arc->weight -= (*small_lm_)(ilabel);
    return true;
  } else {
    return false;
  }
}

float DeltaLmFst::Final(int state_id) const {
  FstArc arc;
  if (lm_->GetArc(state_id, eos_symbol_, &arc)) {
    return lm_->Final(arc.next_state) + arc.weight - (*small_lm_)(eos_symbol_);
  } else {
    return INFINITY;
  }
}

CachedFst::CachedFst(const IFst *fst, int bucket_size) : fst_(fst) {
  buckets_.resize(bucket_size);
  for (std::pair<int, FstArc> &item : buckets_) {
    item.first = -1;
  }
}

int CachedFst::StartState() const {
  return fst_->StartState();
}

float CachedFst::Final(int state) const {
  return fst_->Final(state);
}

bool CachedFst::GetArc(int state, int ilabel, FstArc *arc) {
  // Do nothing when state is 0, we have special optimize for it
  if (state == 0) {
    return fst_->GetArc(state, ilabel, arc);
  }

  int idx = Hash(state, ilabel) % buckets_.size();
  std::pair<int, FstArc> &item = buckets_[idx];
  if (item.first == state && item.second.input_label == ilabel) {
    *arc = item.second;
    return true;
  }

  bool success = fst_->GetArc(state, ilabel, arc);
  if (success) {
    item.first = state;
    item.second = *arc;
  }

  return success;
}

}  // namespace pocketkaldi
