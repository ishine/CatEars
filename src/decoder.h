// decoder/simple-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (author: Arnab Ghoshal);
//                      Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef POCKETKALDI_DECODER_H_
#define POCKETKALDI_DECODER_H_

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include "hashtable.h"
#include "vector.h"
#include "pool.h"
#include "fst.h"
#undef DISALLOW_COPY_AND_ASSIGN
#include "fst/fstlib.h"
#include "am.h"

namespace pocketkaldi {

// Decoder is the core class in pocketkaldi. It decodes the Decodable object
// using viterbi algorithm and stores the best result into Decoder::Hypothesis
// class
//
// This decoder has 2 modes:
//   - Single HCLG mode
//   - Online composition of HCLG and G' mode. In which, G' is a large LM.
//
// In online composition mode, G' should be a backoff LM. ilabel of backoff arc
// should be epsilon (aka 0) and symbols of BOS/EOS (<s> and </s>) should be
// exist.
class Decoder {
 public:
  static constexpr int kBeamSize = 30000;
  static constexpr float kBeamDelta = 0.5;
  static constexpr int kOLabelBeginIdx = -1;
  static constexpr int kNotExist = -1;
  static constexpr int kCutoffSamples = 200;
  static constexpr int kCutoffRandSeed = 0x322;

  // State stores the states of each FST for decoding.
  class State;

  // Stores the decoding result
  class Hypothesis;

  // Initialize the decoder with the FST graph fst. It just borrows the pointer
  // of fst and not own it.
  Decoder(const fst::Fst<fst::StdArc> *fst,
          const Vector<int32_t> &transtion_pdf_id_map,
          float am_scale,
          const DeltaLmFst *delta_lm_fst = nullptr);
  ~Decoder();

  // Initialize decoding and put the root state into beam
  void Initialize();

  // Process and decode current frame (log probability), the best path could be
  // obtain by Decoder::BestPath()
  bool Process(const VectorBase<float> &frame_logp);

  // Change the state to end-of-stream
  void EndOfStream() { is_end_of_stream_ = true; }

  // Get best hypothesis from lattice.
  Hypothesis BestPath();

  // Returns number of frames decoded
  int NumFramesDecoded() const { return num_frames_decoded_; }

 private:
  // Token represents a state in the viterbi lattice. olabel_idx is the index
  // of its corresponded outpu label link-list in the list impl->olabels
  class Token;

  // OLabel is the struct records the list of output labels of a tok in beam. 
  // previous pointes to the previous OLabel like a link list.
  class OLabel;

  // Get the weight cutoff from prev_toks_. We won't go throuth all the toks in
  // beam here to calculate cutoff, Since it takes a long time. Instead, we
  // randomly sample N costs from the beam and GUESS the cutoff value.
  // \param adaptive_beam the gap of cost between best token and the token of
  // beam_size
  // \param best_tokidx index of best token in prev_toks_
  // \return the cutoff of cost to beam size in prev_toks_
  double GetCutoff(float *adaptive_beam, Token **best_tok);

  // Insert tok into self->toks_ with next_state and its output_label. And it
  // will either insert a new token or update existing token in the beam.
  // return true if successfully inserted. Otherwise, when the cost of
  // existing tok is less than new one, return false
  bool InsertTok(
      State next_state, int output_label, OLabel *prev_olabel, float cost);

  // Processes nonemitting arcs for one frame. Propagates within cur_toks_.
  void ProcessNonemitting(double cutoff);

  // Process the emitting (non-epsilon) arcs of each states in the beam
  // \return cutoff of next weight
  float ProcessEmitting(const VectorBase<float> &frame_logp);

  // Propogate the lm_state with ilabel in DeltaLmFst. Return the next state
  // in DeltaLmFst and set weight to the cost of transition.
  int32_t PropogateLm(int32_t lm_state, int ilabel, float *weight);

  // Get log likelihood of transition-id in current frame
  float LogLikelihood(const VectorBase<float> &frame_logp, int trans_id) const;

  // Only used in get_cutoff()
  std::vector<float> costs_;

  // FST graph used for decoding
  const fst::Fst<fst::StdArc> *fst_;

  // Additional graph F = G^{-1} o G', where G^{-1} is the same as G in HCLG
  // graph except that all the weights are negative. G' is a big language model
  std::unique_ptr<CachedFst> delta_lm_fst_;

  // Frames decoded
  int num_frames_decoded_;

  // Tokens used in decoding. toks_ is the current beam, all generated tokens
  // will be placed here. And after frame advanced, the toks_ will be swapped
  // with prev_toks_
  Pool<Token> toks_pool_;
  std::vector<Token *> toks_;
  std::vector<Token *> prev_toks_;

  // Stores the map between state-id and the index of corresponded token in
  // toks_
  HashTable<State, int32_t> state_idx_;

  // Storea all output-label nodes
  GCPool<OLabel> olabels_pool_;

  // Scale for AM
  float am_scale_;

  // true if current stream is end
  bool is_end_of_stream_;

  // Map from transition-id pdf-id
  const Vector<int32_t> &transtion_pdf_id_map_;

  // Beam threshold
  float beam_;
};


// Stores the state of each FST
class Decoder::State {
 public:
  State(int32_t hclg_state, int32_t lm_state);
  State();

  int32_t hclg_state() const { return hclg_state_; }
  int32_t lm_state() const { return lm_state_; }

  bool operator==(const State &s) const {
    return hclg_state_ == s.hclg_state_ && lm_state_ == s.lm_state_;
  }
 
 private:
  int32_t hclg_state_;
  int32_t lm_state_;
};

// Hash fucntions for state
inline int32_t hash(Decoder::State s) {
  int32_t h = 19;
  h = h * 31 + s.hclg_state();
  h = h * 31 + s.lm_state();

  return h;
}

inline std::string ToString(const Decoder::State &state) {
  return util::Format("State({}, {})", state.hclg_state(), state.lm_state());
}

// Stores the decoding result
class Decoder::Hypothesis {
 public:
  Hypothesis(const std::vector<int> &words, float weight);

  // Word-ids in the decoding result
  const std::vector<int> &words() const { return words_; }

  // Weight for this utterance
  float weight() const { return weight_; }

 private:
  std::vector<int> words_;  
  float weight_;
};

class Decoder::Token {
 public:
  Token(State state, float cost, OLabel *olabel);

  // The state in FST
  State state() const { return state_; }

  // Current cost
  float cost() const { return cost_; }

  // Head of output label chain
  OLabel *olabel() const { return olabel_; }

 private:
  OLabel *olabel_;
  State state_;
  float cost_;
};

class Decoder::OLabel : public Collectable {
 public:
  OLabel(OLabel *previous, int olabel);

  // Called when LmToken is garbage collected
  void OnCollect() override;

  // Index of previous olabel
  OLabel *previous() const { return previous_; }

  // Output label of current node
  int olabel() const { return olabel_; }

  // Next state with input label. Return nullptr when next state with ilabel
  // not exist 
  OLabel *next(int ilabel) const {
    std::unordered_map<int, OLabel *>::const_iterator it = nexts_.find(ilabel);
    if (it != nexts_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }
  void set_next(int ilabel, OLabel *next) {
    nexts_[ilabel] = next;
  }

 private:
  OLabel *previous_;
  int olabel_;
  std::unordered_map<int, OLabel *> nexts_;
};

}  // namespace pocketkaldi


#endif  // POCKETKALDI_DECODER_H_
