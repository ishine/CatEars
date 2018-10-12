// Created at 2016-11-24

#ifndef POCKETKALDI_FST_H_
#define POCKETKALDI_FST_H_

#include <stdint.h>
#include <string>
#include <utility>
#include "util.h"
#include "status.h"
#include "pocketkaldi.h"
#include "vector.h"

namespace pocketkaldi {

struct FstArc {
  int32_t next_state;
  int32_t input_label;
  int32_t output_label;
  float weight;

  FstArc();
  FstArc(int next_state, int ilabel, int olabel, float weight);
};

// Interface for Fst
class IFst {
 public:
  // Get the start state of this Fst
  virtual int StartState() const = 0;

  // Get out-going arc of state with ilabel. On success return true. If arc with
  // specific ilabel not exist, return false 
  virtual bool GetArc(int state, int ilabel, FstArc *arc) const = 0;

  // Get the final score of state. If the state is non-terminal, returns 0
  virtual float Final(int state_id) const = 0;
};

class Fst : public IFst {
public:
  // Iterators of out-going arcs for a state
  class ArcIterator {
   public:
    ArcIterator(int base, int total, const FstArc *arcs);
    ~ArcIterator();

    // If next arc exists, retrun it and move the iterator forward, else return
    // nullptr
    const FstArc *Next();

   private:
    int base_;
    int cnt_pos_;
    int total_;
    const FstArc *arcs_;
  };

  // Consts
  static const char *kSectionName;
  static constexpr int kNoState = -1;

  Fst();
  virtual ~Fst();

  // Read fst from binary file.
  Status Read(util::ReadableFile *fd);

  // Start state of this Fst
  int StartState() const override;

  // Get out-going arc of state with ilabel. On success return true. If arc with
  // specific ilabel not exist, return false 
  bool GetArc(int state, int ilabel, FstArc *arc) const override;

  // Get the final score of state. If the state is non-terminal, returns 0
  float Final(int state_id) const override;

  // Iterate out-going arcs for a state
  ArcIterator IterateArcs(int state) const;

  // Return the type of this fst
  std::string fst_type() const { return fst_type_; }

 protected:
  // Calcuate the number of outcoming arcs for state
  int CountArcs(int state) const;

  int start_state_;
  std::string fst_type_;
  std::vector<FstArc> arcs_;
  std::vector<int32_t> state_idx_;
  std::vector<float> final_;
};

// Fst for language model, including deterministic on demand for back-off arcs,
class LmFst : public Fst {
 public:
  static constexpr char kLmFst[] = "pk::fst_lm";

  // Get out-going arc of state with ilabel. For LM, we will follow the back-off
  // arc automatically when there is no arc of ilabel in current state
  // If no matched arc even follow the back-off arc, return false. 
  bool GetArc(int state, int ilabel, FstArc *arc) const override;

  // Get the final score of state. If current state is not a final state. It
  // will flollow the back-off arc automatically
  float Final(int state_id) const override;

  // Initialize bucket for state 0 optimization
  void InitBucket0();

 private:
  // Get the backoff arc for given state. If there is no back-off arc return
  // nullptr
  const FstArc *GetBackoffArc(int state) const;

  std::vector<FstArc> bucket_0_;
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
class DeltaLmFst : public IFst {
 public:
  DeltaLmFst(const Vector<float> *small_lm,
             const LmFst *lm,
             const SymbolTable *symbol_table);

  // Start state of this Fst. It will transduce the <s> symbol and return the
  // state as start state.
  // Here we assuming weight of arc from start_state with input symbol <s> is
  // zero  
  int StartState() const override;

  // Find the arc in small_lm_ and minus the weight from small_lm_
  bool GetArc(int state, int ilabel, FstArc *arc) const override;

  // Get the final score from lm_ then minus the </s> weight in small_lm_. 
  float Final(int state_id) const override;

 private:
  const Vector<float> *small_lm_;
  const LmFst *lm_;

  int bos_symbol_;
  int eos_symbol_;
};


// Provice cache for GetArc method
class CachedFst {
 public:
  explicit CachedFst(const IFst *fst, int bucket_size);

  // Implement interface IFst
  int StartState() const;

  // Implement interface IFst
  bool GetArc(int state, int ilabel, FstArc *arc);

  // Implement interface IFst 
  float Final(int state_id) const;

 private:
  std::vector<std::pair<int, FstArc>> buckets_;
  const IFst *fst_;

  // Compute hash value for state and ilabel
  inline int32_t Hash(int state, int ilabel) const {
    int32_t h = 19;
    h = h * 31 + state;
    h = h * 31 + ilabel;

    return h;
  }
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_FST_H_