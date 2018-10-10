// Created at 2018-09-28

#ifndef POCKETKALDI_LM_FST_H_
#define POCKETKALDI_LM_FST_H_

#include "fst.h"
#include "vector.h"

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

// DeltaLmFst is the composition of G^{-1} and G'. Where G^{-1} has the negative
// weights of G in HCLG fst. And G' is a big LM.
// Here we assuming that G is just a unigram language model, so we don't need to
// store G^{-1} as a FST, we just store the weight of each word into a vector.
//
// Here we also assuming lm_ is a backoff lm fst with BOS and EOS symbols. And
// DeltaLmFst will transduce <s> and </s> symbol automatically when calling
// StartState() and Final(). To make it looks like a LM fst without EOS/BOS
// symbols
class DeltaLmFst {
 public:
  DeltaLmFst(const Vector<float> *small_lm_,
             const LmFst *lm_,
             const SymbolTable *symbol_table);

  // Start state of this Fst. It will transduce the <s> symbol and return the
  // state as start state.
  // Here we assuming weight of arc from start_state with input symbol <s> is
  // zero  
  int StartState() const;

  // Find the arc in small_lm_ and minus the weight from small_lm_
  bool GetArc(int state, int ilabel, Fst::Arc *arc) const;

  // Get the final score from lm_ then minus the </s> weight in small_lm_. 
  float Final(int state_id) const;

 private:
  const Vector<float> *small_lm_;
  const LmFst *lm_;

  int bos_symbol_;
  int eos_symbol_;
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_FST_H_
