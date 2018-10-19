// Created at 2017-04-12

#include "decoder.h"

#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include "hashtable.h"
#include "fst.h"

namespace pocketkaldi {

Decoder::State::State(int32_t hclg_state, int32_t lm_state):
    hclg_state_(hclg_state),
    lm_state_(lm_state) {}

Decoder::State::State(): hclg_state_(0), lm_state_(0) {}

Decoder::Token::Token(State state, float cost, OLabel *olabel):
    state_(state),
    cost_(cost),
    olabel_(olabel) {
}

Decoder::OLabel::OLabel(OLabel *previous, int olabel):
    previous_(previous),
    olabel_(olabel) {
}

void Decoder::OLabel::OnCollect() {
  nexts_.clear();

  // Remove itself from previous_->nexts_ if previous_ is not freed
  if (previous_) {
    if (!previous_->is_freed()) {
      previous_->nexts_.erase(olabel_);
    }
    previous_ = nullptr;
  }

  olabel_ = -1;
}

Decoder::Hypothesis::Hypothesis(const std::vector<int> &words, float weight):
    words_(words),
    weight_(weight) {
}
Decoder::Decoder(
    const Fst *fst,
    const Vector<int32_t> &transtion_pdf_id_map,
    float am_scale,
    const DeltaLmFst *delta_lm_fst):
        fst_(fst),
        beam_(16.0),
        state_idx_(kBeamSize * 4),
        transtion_pdf_id_map_(transtion_pdf_id_map),
        am_scale_(am_scale),
        is_end_of_stream_(false) {
  if (delta_lm_fst) {
    delta_lm_fst_ = std::unique_ptr<CachedFst>(
        new CachedFst(delta_lm_fst, 1000000));
  }
}

Decoder::~Decoder() {
  fst_ = nullptr;
}

bool Decoder::Process(const VectorBase<float> &frame_logp) {
  PK_DEBUG(util::Format("frame: {}", num_frames_decoded_));

  double cutoff = ProcessEmitting(frame_logp);
  if (!std::isfinite(cutoff)) return false;
  
  ProcessNonemitting(cutoff);

  // Exit when there is no active tokens
  if (toks_.size() == 0) return false;

  // GC of olabel nodes
  if (num_frames_decoded_ % 20 == 0) {
    double free_nodes = olabels_pool_.free_nodes();
    double allocated_nodes = olabels_pool_.allocated_nodes();

    std::vector<OLabel *> olabel_root;
    for (Token *tok : toks_) {
      if (tok->olabel()) olabel_root.push_back(tok->olabel());
    }
    olabels_pool_.GC(olabel_root);
  }

  num_frames_decoded_++;
  return true;
}

float Decoder::LogLikelihood(const VectorBase<float> &frame_logp,
                             int trans_id) const {
  int pdf_id = transtion_pdf_id_map_(trans_id);
  float logp = frame_logp(pdf_id);
  return am_scale_ * logp;
}

void Decoder::Initialize() {
  // Prepare beams
  toks_.clear();
  prev_toks_.clear();

  // Initialize decoding:
  int start_state = fst_->StartState();
  assert(start_state >= 0);

  int lm_start_state = 0;
  if (delta_lm_fst_) {
    lm_start_state = delta_lm_fst_->StartState();
  }

  InsertTok(State(start_state, lm_start_state), 0, nullptr, 0.0f);
  num_frames_decoded_ = 0;
  ProcessNonemitting(INFINITY);
}

int32_t Decoder::PropogateLm(int32_t lm_state, int ilabel, float *weight) {
  assert(delta_lm_fst_ != nullptr);
  FstArc delta_lm_arc;

  if (ilabel != 0) {
    bool success = delta_lm_fst_->GetArc(lm_state,
                                         ilabel,
                                         &delta_lm_arc);
    if (success) {
      *weight = delta_lm_arc.weight;
      return delta_lm_arc.next_state;
    } else {
      PK_WARN("decoder: HCLG output and LM input symbol mismatch");
    }
  }

  *weight = 0.0f;
  return lm_state;
}

bool Decoder::InsertTok(
    State next_state,
    int output_label,
    OLabel *prev_olabel,
    float cost) {
  // PK_DEBUG(util::Format("insert state = {}", next_state));
  int tok_idx = state_idx_.Find(next_state, kNotExist);
  
  // Create the olabel for next tok when the output_label of arc
  // is not 0 (epsilon)
  OLabel *next_olabel = nullptr;
  if (output_label != 0) {
    if (prev_olabel) next_olabel = prev_olabel->next(output_label);
    if (!next_olabel) {
      next_olabel = olabels_pool_.Alloc(prev_olabel, output_label);
      if (prev_olabel) prev_olabel->set_next(output_label, next_olabel);
    }
  } else {
    next_olabel = prev_olabel;
  }

  // Insert new or update existing token in the beam
  if (tok_idx == kNotExist) {
    int num_toks = toks_.size();
    toks_.push_back(toks_pool_.Alloc(next_state, cost, next_olabel));
    state_idx_.Insert(next_state, num_toks);
  } else {
    // If the cost of existing token is less than the new one, just discard
    // inserting and return false
    if (toks_[tok_idx]->cost() > cost) {
      toks_[tok_idx] = toks_pool_.Alloc(next_state, cost, next_olabel);
    } else {
      return false;
    }
  }
  return true;
}

double Decoder::GetCutoff(float *adaptive_beam, Token **best_tok) {
  double best_cost = INFINITY;
  *best_tok = prev_toks_[0];

  costs_.clear();
  uint64_t next_random = kCutoffRandSeed;

  // Probability of sample a cost into self->costs
  float sample_prob = kCutoffSamples / (float)prev_toks_.size();
  
  for (Token *tok : prev_toks_) {
    // Random sample costs from beam. To be consistant even in multi-thread
    // environment, do don't use rand() here
    next_random = next_random * (uint64_t)25214903917 + 11;
    float random_f = (next_random & 0xffff) / (float)65535;
    if (random_f < sample_prob) {
      costs_.push_back(tok->cost());
    }

    if (tok->cost() < best_cost) {
      best_cost = tok->cost();
      *best_tok = tok;
    }
  }

  // Exit if NAN found
  if (!std::isfinite(best_cost)) return INFINITY;

  double beam_cutoff = best_cost + beam_;
  double max_active_cutoff = NAN;
  
  // Here we guess the cutoff weight to limit the prev_toks_ to beam size
  if (prev_toks_.size() > kBeamSize) {
    int cutoff_idx = costs_.size() * kBeamSize / prev_toks_.size();
    std::nth_element(costs_.begin(), costs_.begin() + cutoff_idx, costs_.end());
    PK_DEBUG(util::Format(
        "Cutoff: total = {}, cutoff_idx = {}, cutoff = {},{},{},{}",
        costs_.size(),
        cutoff_idx,
        costs_[cutoff_idx - 3],
        costs_[cutoff_idx - 2],
        costs_[cutoff_idx - 1],
        costs_[cutoff_idx]));
    max_active_cutoff = costs_[cutoff_idx];
  }

  if (max_active_cutoff < beam_cutoff) { 
    // max_active_cutoff is tighter than beam.
    *adaptive_beam = max_active_cutoff - best_cost + kBeamDelta;
    beam_cutoff = max_active_cutoff;
  } else {
    *adaptive_beam = beam_;
  }

  return beam_cutoff;
}


// Processes nonemitting arcs for one frame.  Propagates within cur_toks_.
void Decoder::ProcessNonemitting(double cutoff) {
  PK_DEBUG("ProcessNonemitting()");
  std::vector<State> queue;
  for (const Token *tok : toks_) {
    assert(tok->state().hclg_state() >= 0);
    // PK_DEBUG(util::Format("queue->push_back({})", tok->state()));
    queue.push_back(tok->state());
  }

  // Loop until no state in beam have out-going epsilon arc
  while (!queue.empty()) {
    State state = queue.back();
    queue.pop_back();

    // Get tok by state
    // PK_DEBUG(util::Format("state_idx_.Find({}, kNotExist)", state));
    int tok_idx = state_idx_.Find(state, kNotExist);
    assert(tok_idx != kNotExist);

    Fst::ArcIterator arc_iter = fst_->IterateArcs(state.hclg_state());
    const FstArc *arc = nullptr;
    while ((arc = arc_iter.Next()) != nullptr) {
      // propagate nonemitting only...
      if (arc->input_label != 0) continue;

      const float ac_cost = 0.0;
      const Token *from_tok = toks_[tok_idx];
      double total_cost = from_tok->cost() + arc->weight + ac_cost;

      // Online compose with G' when available
      State state = from_tok->state();
      int32_t lm_state = state.lm_state();
      if (delta_lm_fst_) {
        float lm_weight = 0.0f;
        lm_state = PropogateLm(lm_state, arc->output_label, &lm_weight);
        total_cost += lm_weight;
      }

      if (total_cost > cutoff) continue;

      // Create and insert tok into beam
      // If the token successfully inserted or updated in the beam, `inserted`
      // will be true and then we will push the new state into `queue`
      bool inserted = InsertTok(
          State(arc->next_state, lm_state),
          arc->output_label,
          from_tok->olabel(),
          total_cost);
      if (inserted) queue.push_back(State(arc->next_state, lm_state));
    }
  }
}

// Process the emitting (non-epsilon) arcs of each states in the beam
float Decoder::ProcessEmitting(const VectorBase<float> &frame_logp) {
  PK_DEBUG("ProcessEmitting()");
  // Clear the prev_toks_
  state_idx_.Clear();

  // Swap toks_ and empty prev_toks_
  PK_DEBUG(util::Format("toks_.size() = {}", toks_.size()));
  toks_.swap(prev_toks_);

  // Calculate beam_cutoff of beam
  float adaptive_beam = INFINITY;
  Token *best_tok = 0;
  float weight_cutoff = GetCutoff(&adaptive_beam, &best_tok);
  PK_DEBUG(util::Format("weight_cutoff = {}", weight_cutoff));
  PK_DEBUG(util::Format("adaptive_beam = {}", adaptive_beam));
  if (!std::isfinite(weight_cutoff)) return INFINITY;

  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff = INFINITY;

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  State best_state = best_tok->state();
  PK_DEBUG(util::Format("best_state = {}", best_state));
  Fst::ArcIterator arc_iter = fst_->IterateArcs(best_state.hclg_state());
  const FstArc *arc = nullptr;
  while ((arc = arc_iter.Next()) != nullptr) {
    if (arc->input_label == 0) continue;

    float acoustic_cost = -LogLikelihood(frame_logp, arc->input_label);
    double total_cost = best_tok->cost() + arc->weight + acoustic_cost;

    // Online compose with G' when available
    if (delta_lm_fst_) {
      float lm_weight = 0.0f;
      PropogateLm(best_state.lm_state(), arc->output_label, &lm_weight);
      total_cost += lm_weight;
    }

    if (total_cost + adaptive_beam < next_weight_cutoff) {
      next_weight_cutoff = total_cost + adaptive_beam;
    }
  }
  
  // Ok, we iterate each token in prev_tok_ and add new tokens into toks_ with
  // the emitting arcs of them.
  for (Token *from_tok : prev_toks_) {
    State state = from_tok->state();

    // weight_cutoff is computed according to beam size
    // So there are only top beam_size toks less than weight_cutoff
    if (from_tok->cost() > weight_cutoff) continue;

    Fst::ArcIterator arc_iter = fst_->IterateArcs(state.hclg_state());
    const FstArc *arc = nullptr;
    while ((arc = arc_iter.Next()) != nullptr) {
      if (arc->input_label == 0) continue;

      float ac_cost = -LogLikelihood(frame_logp, arc->input_label);
      double total_cost = from_tok->cost() + arc->weight + ac_cost;

      // Online compose with G' when available
      int32_t lm_state = state.lm_state();
      if (delta_lm_fst_) {
        float lm_weight = 0.0f;
        lm_state = PropogateLm(state.lm_state(), arc->output_label, &lm_weight);
        total_cost += lm_weight;
      }
      
      // Prune the toks whose cost is too high
      if (total_cost > next_weight_cutoff) continue;
      if (total_cost + adaptive_beam < next_weight_cutoff) {
        next_weight_cutoff = total_cost + adaptive_beam;
      }

      // Create and insert the tok into toks_
      assert(arc->next_state >= 0 && lm_state >= 0);
      InsertTok(
          State(arc->next_state, lm_state),
          arc->output_label,
          from_tok->olabel(),
          total_cost);
    }

    toks_pool_.Dealloc(from_tok);
  }

  // All pointers in prev_toks_ is freed
  prev_toks_.clear();

  return next_weight_cutoff;
}

// Get best hypothesis from lattice.
Decoder::Hypothesis Decoder::BestPath() {
  std::vector<int> words;
  float weight;

  // Find the best token
  int best_idx = kNotExist;
  double best_cost = INFINITY;
  for (int i = 0; i < toks_.size(); ++i) {
    Token *tok = toks_[i];
    State state = tok->state();
    double cost = tok->cost();

    if (is_end_of_stream_) {
      cost += fst_->Final(state.hclg_state());
    }
    if (delta_lm_fst_ && is_end_of_stream_) {
      float lm_weight = delta_lm_fst_->Final(tok->state().lm_state());
      cost += lm_weight;
    }

    if (cost != INFINITY && cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
  }

  if (best_idx == kNotExist) return Hypothesis(std::vector<int>(), 0.0f);

  // Get all output labels from best_tok
  Token *best_tok = toks_[best_idx];
  PK_DEBUG(util::Format("best_tok.state = {}", best_tok->state()));
  PK_DEBUG(util::Format("best_tok.cost = {}", best_tok->cost()));
  OLabel *best_olabel = best_tok->olabel();
  OLabel *olabel = best_olabel;
  while (olabel != nullptr) {
    PK_DEBUG(util::Format("Word: {}", olabel->olabel()));
    words.push_back(olabel->olabel());
    olabel = olabel->previous();
    assert(words.size() < 100000);
  }

  weight = best_cost;
  weight += fst_->Final(best_tok->state().hclg_state());

  return Hypothesis(words, weight);
}

}  // namespace pocketkaldi
