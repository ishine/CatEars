// Created at 2016-11-24

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <functional>
#include "fst.h"
#include "util.h"
#include "symbol_table.h"

using pocketkaldi::Status;
using pocketkaldi::SymbolTable;
using pocketkaldi::Fst;
using pocketkaldi::FstArc;
using pocketkaldi::LmFst;
using pocketkaldi::DeltaLmFst;
using pocketkaldi::Vector;
using pocketkaldi::util::ReadableFile;
using pocketkaldi::util::Split;
using pocketkaldi::util::Format;
using pocketkaldi::util::StringToLong;

void TestFst() {
  Status status;
  ReadableFile fd;
  status = fd.Open(TESTDIR "data/testinput.fst");
  assert(status.ok());


  Fst fst;
  status = fst.Read(&fd);
  assert(status.ok());

  assert(fst.StartState() == 0);
  assert(fst.Final(0) == std::numeric_limits<double>::infinity());
  assert(fst.Final(1) == std::numeric_limits<double>::infinity());
  assert(fst.Final(2) == 3.5f);

  
  const FstArc *arc = nullptr;

  Fst::ArcIterator arc_iter = fst.IterateArcs(0);
  arc = arc_iter.Next();
  assert(arc);
  assert(arc->next_state == 1);
  assert(arc->input_label == 1);
  assert(arc->output_label == 1);
  assert(arc->weight == 0.5f);
  arc = arc_iter.Next();
  assert(arc);
  assert(arc->next_state == 1);
  assert(arc->input_label == 2);
  assert(arc->output_label == 2);
  assert(arc->weight == 1.5f);
  arc = arc_iter.Next();
  assert(arc == nullptr);

  arc_iter = fst.IterateArcs(1);
  arc = arc_iter.Next();
  assert(arc);
  assert(arc->next_state == 2);
  assert(arc->input_label == 3);
  assert(arc->output_label == 3);
  assert(arc->weight == 2.5f);
  arc = arc_iter.Next();
  assert(arc == nullptr);

  arc_iter = fst.IterateArcs(2);
  arc = arc_iter.Next();
  assert(arc == nullptr);
}

// Convert words to word-ids
std::vector<int> ConvertToWordIds(const std::vector<std::string> &words,
                                  const SymbolTable &symbol_table) {
  std::vector<int> word_ids;
  for (const std::string &word : words) {
    int word_id = symbol_table.GetId(word);
    assert(word_id != SymbolTable::kNotExist && "unexpected word");

    word_ids.push_back(word_id);
  }

  return word_ids;
}

// Get the lm score of given query using FST
float LmScore(const LmFst &lm_fst,
              const SymbolTable &symbol_table,
              const std::string &query) {
  std::vector<std::string> words = Split(query, " ");
  std::vector<int> word_ids = ConvertToWordIds(words, symbol_table);

  float score = 0;
  int start_state = lm_fst.StartState();
  printf("start_state = %d, score = %f\n", start_state, score);
  FstArc arc;

  // BOS
  assert(lm_fst.GetArc(start_state, symbol_table.bos_id(), &arc));
  int state = arc.next_state;
  score += arc.weight;
  printf("bos_state = %d, score = %f\n", state, score);

  for (int word_id : word_ids) {
    printf("word_id = %d\n", word_id);
    assert(lm_fst.GetArc(state, word_id, &arc));
    state = arc.next_state;
    score += arc.weight;
    printf("state = %d, score = %f\n", state, score);
  }

  // EOS
  assert(lm_fst.GetArc(state, symbol_table.eos_id(), &arc));
  score += arc.weight;
  state = arc.next_state;
  printf("eos_state = %d, score = %f\n", state, score);

  // Final
  score += lm_fst.Final(state);

  return -score;
}

// Get the lm score of given query using FST
float DeltaLmScore(const DeltaLmFst &delta_lm_fst,
                   const SymbolTable &symbol_table,
                   const std::string &query) {
  std::vector<std::string> words = Split(query, " ");
  std::vector<int> word_ids = ConvertToWordIds(words, symbol_table);

  float score = 0;
  int state = delta_lm_fst.StartState();
  printf("start_state = %d, score = %f\n", state, score);
  FstArc arc;

  for (int word_id : word_ids) {
    printf("word_id = %d\n", word_id);
    assert(delta_lm_fst.GetArc(state, word_id, &arc));
    state = arc.next_state;
    score += arc.weight;
    printf("state = %d, score = %f\n", state, score);
  }

  // Final
  score += delta_lm_fst.Final(state);
  printf("final: score = %f\n", score);

  return score;
}

void TestLmFst() {
  LmFst lm_fst;
  ReadableFile fd_fst;
  Status status = fd_fst.Open(TESTDIR "data/G.pfst");
  assert(status.ok());

  status = lm_fst.Read(&fd_fst);
  assert(status.ok());

  SymbolTable symbol_table;
  status = symbol_table.Read(TESTDIR "data/lm.words.txt");
  assert(status.ok());

  // check_query checks if lm_score of query matches parameter score
  std::function<bool(float, const std::string&)>
  check_query = [&] (float score, const std::string &query) {
    return fabs(score - LmScore(lm_fst, symbol_table, query)) < 1e-5;
  };

  assert(check_query(-38.767048, "marisa runs the kirisame magic shop"));
  assert(check_query(-28.481011, "reimu and marisa are friends"));
  assert(check_query(-62.663559, "reimu and marisa are playable characters in the games of touhou"));
  assert(check_query(-6.2797366, "marisa"));
}

void TestDeltaLmFst() {
  ReadableFile fd_small_lm;
  Status status = fd_small_lm.Open(TESTDIR "data/lm.1order.bin");
  assert(status.ok());
  Vector<float> small_lm;
  status = small_lm.Read(&fd_small_lm);
  assert(status.ok());

  ReadableFile fd_fst;
  LmFst lm_fst;
  status = fd_fst.Open(TESTDIR "data/G.pfst");
  assert(status.ok());
  status = lm_fst.Read(&fd_fst);
  assert(status.ok());

  SymbolTable symbol_table;
  status = symbol_table.Read(TESTDIR "data/lm.words.txt");
  assert(status.ok());

  DeltaLmFst delta_lm_fst(&small_lm, &lm_fst, &symbol_table);

  // check_query checks if lm_score of query matches parameter score
  std::function<bool(float, const std::string&)>
  check_query = [&] (float score, const std::string &query) {
    float delta_score = DeltaLmScore(delta_lm_fst, symbol_table, query);
    return fabs(score - delta_score) < 1e-5;
  };

  assert(check_query(0.886695, "marisa runs the kirisame magic shop"));
  assert(check_query(-1.433023, "reimu and marisa are friends"));
  assert(check_query(-0.688201, "reimu and marisa are playable characters in the games of touhou"));
  assert(check_query(-0.510554, "marisa"));
}

int main() {
  TestFst();
  TestLmFst();
  TestDeltaLmFst();

  return 0;
}
