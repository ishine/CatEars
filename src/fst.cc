// Created at 2016-11-24

#include "fst.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <array>
#include <algorithm>

namespace pocketkaldi {

const char *Fst::kSectionName = "pk::fst_0";

Fst::Arc::Arc() {}
Fst::Arc::Arc(int next_state, int ilabel, int olabel, float weight): 
    next_state(next_state),
    input_label(ilabel),
    output_label(olabel),
    weight(weight) {}


Fst::Fst(): start_state_(0) {}
Fst::~Fst() {}

Fst::ArcIterator::ArcIterator(int base, int total, const Arc *arcs) :
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
      arc_number * sizeof(Arc);
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

bool Fst::GetArc(int state, int ilabel, Arc *arc) const {
  int num_arcs = CountArcs(state);
  if (num_arcs == 0) return false;

  int start_idx = state_idx_[state];
  const Arc *first = &arcs_[start_idx];
  const Arc *last = &arcs_[start_idx + num_arcs];


  const Arc *found_arc = std::lower_bound(
      first,
      last,
      Arc(0, ilabel, 0, 0.0f), 
      [] (const Fst::Arc &l, const Fst::Arc &r) {
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

const Fst::Arc *Fst::ArcIterator::Next() {
  if (cnt_pos_ < total_) {
    const Arc *arc = &arcs_[base_ + cnt_pos_];
    ++cnt_pos_;
    return arc;
  } else {
    return nullptr;
  }
}

}  // namespace pocketkaldi
