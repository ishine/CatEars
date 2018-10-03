// Created at 2016-11-24

#ifndef POCKETKALDI_FST_H_
#define POCKETKALDI_FST_H_

#include <stdint.h>
#include <string>
#include "util.h"
#include "status.h"
#include "pocketkaldi.h"

namespace pocketkaldi {

class Fst {
public:
  // An out-going arc in Fst
  struct Arc {
    int32_t next_state;
    int32_t input_label;
    int32_t output_label;
    float weight;

    Arc();
    Arc(int next_state, int ilabel, int olabel, float weight);
  };

  // Iterators of out-going arcs for a state
  class ArcIterator {
   public:
    ArcIterator(int base, int total, const Arc *arcs);
    ~ArcIterator();

    // If next arc exists, retrun it and move the iterator forward, else return
    // nullptr
    const Arc *Next();

   private:
    int base_;
    int cnt_pos_;
    int total_;
    const Arc *arcs_;
  };

  // Consts
  static const char *kSectionName;
  static constexpr int kNoState = -1;

  Fst();
  virtual ~Fst();

  // Read fst from binary file.
  Status Read(util::ReadableFile *fd);

  // Start state of this Fst
  int start_state() const {
    return start_state_;
  }

  // Get out-going arc of state with ilabel. On success return true. If arc with
  // specific ilabel not exist, return false 
  virtual bool GetArc(int state, int ilabel, Arc *arc) const;

  // Get the final score of state. If the state is non-terminal, returns 0
  virtual float Final(int state_id) const {
    assert(state_id < final_.size());
    return final_[state_id];
  }


  // Iterate out-going arcs for a state
  ArcIterator IterateArcs(int state) const;

  // Return the type of this fst
  std::string fst_type() const { return fst_type_; }

 protected:
  // Calcuate the number of outcoming arcs for state
  int CountArcs(int state) const;

  int start_state_;
  std::string fst_type_;
  std::vector<Arc> arcs_;
  std::vector<int32_t> state_idx_;
  std::vector<float> final_;
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_FST_H_