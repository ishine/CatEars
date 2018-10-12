// Created at 2017-03-29

#include "pocketkaldi.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <string>
#include "am.h"
#include "cmvn.h"
#include "decodable.h"
#include "decoder.h"
#include "fbank.h"
#include "fst.h"
#include "nnet.h"
#include "symbol_table.h"
#include "pcm_reader.h"
#include "configuration.h"
#include "util.h"

using pocketkaldi::Decoder;
using pocketkaldi::Fbank;
using pocketkaldi::CMVN;
using pocketkaldi::util::ToRawStatus;
using pocketkaldi::Status;
using pocketkaldi::Configuration;
using pocketkaldi::Vector;
using pocketkaldi::VectorBase;
using pocketkaldi::Matrix;
using pocketkaldi::SubVector;
using pocketkaldi::AcousticModel;
using pocketkaldi::Decodable;
using pocketkaldi::SymbolTable;
using pocketkaldi::LmFst;
using pocketkaldi::DeltaLmFst;

// The internal version of an utterance. It stores the intermediate state in
// decoding.
typedef struct pk_utterance_internal_t {
  pocketkaldi::Vector<float> raw_wave;
} pk_utterance_internal_t;

void pk_init(pk_t *self) {
  self->fst = NULL;
  self->am = NULL;
  self->cmvn_global_stats = NULL;
  self->symbol_table = NULL;
  self->fbank = NULL;
  self->enable_cmvn = 0;
  self->delta_lm_fst = nullptr;
  self->large_lm_fst = nullptr;
  self->original_lm = nullptr;
}

void pk_destroy(pk_t *self) {
  delete self->fst;
  self->fst = nullptr;

  delete self->am;
  self->am = NULL;

  delete self->cmvn_global_stats;
  self->cmvn_global_stats = nullptr;

  delete self->symbol_table;
  self->symbol_table = nullptr;

  delete self->fbank;
  self->fbank = NULL;

  self->enable_cmvn = 0;
}

// Read CMVN 
Status ReadCMVN(pk_t *self, const Configuration &conf) {
  // Get cmvn_stats filename from config file
  std::string filename = conf.GetPathOrElse("cmvn_stats", "");
  if (filename == "") {
    return pocketkaldi::Status::Corruption(pocketkaldi::util::Format(
        "Unable to find key 'cmvn_stats' in {}",
        conf.filename()));
  }

  pocketkaldi::util::ReadableFile fd;
  PK_CHECK_STATUS(fd.Open(filename));
  self->cmvn_global_stats = new Vector<float>();
  PK_CHECK_STATUS(self->cmvn_global_stats->Read(&fd));


  // Read flag of CMVN
  int enable_cmvn = 0;
  PK_CHECK_STATUS(conf.GetInteger("enable_cmvn", &enable_cmvn));
  self->enable_cmvn = enable_cmvn;

  return Status::OK();
}

// Read symbol table
Status ReadSymbolTable(pk_t *self, const Configuration &conf) {
  std::string filename = conf.GetPathOrElse("symbol_table", "");
  if (filename == "") {
    return pocketkaldi::Status::Corruption(pocketkaldi::util::Format(
        "Unable to find key 'symbol_table' in {}",
        filename));
  }
  self->symbol_table = new SymbolTable();
  PK_CHECK_STATUS(self->symbol_table->Read(filename)); 

  return Status::OK();
}

// Read and initialize delta lm fst
Status ReadDeltaLmFst(pk_t *self, const Configuration &conf) {
  std::string large_lm_file = conf.GetPathOrElse("large_lm", "");

  // If delta_lm_fst is not enables
  if (large_lm_file == "") return Status::OK();

  // Origianl LM in HCLG
  std::string original_lm_file = conf.GetPathOrElse("original_lm", "");
  if (original_lm_file == "") {
    return pocketkaldi::Status::Corruption(pocketkaldi::util::Format(
        "Unable to find key 'original_lm' in {}",
        original_lm_file));
  }

  pocketkaldi::util::ReadableFile fd_original_lm;
  PK_CHECK_STATUS(fd_original_lm.Open(original_lm_file));
  self->original_lm = new Vector<float>();
  PK_CHECK_STATUS(self->original_lm->Read(&fd_original_lm));

  // Large LM
  pocketkaldi::util::ReadableFile fd_large_lm;
  PK_CHECK_STATUS(fd_large_lm.Open(large_lm_file));
  self->large_lm_fst = new LmFst();
  PK_CHECK_STATUS(self->large_lm_fst->Read(&fd_large_lm));

  // Build DeltaLmFst
  assert(self->symbol_table != nullptr);
  self->delta_lm_fst = new DeltaLmFst(self->original_lm,
                                      self->large_lm_fst,
                                      self->symbol_table);

  return Status::OK();
}

// Pocketkaldi model struct
//   FST
//   CMVN
//   TRANS_MODEL
//   AM
//   SYMBOL_TABLE
void pk_load(pk_t *self, const char *filename, pk_status_t *status) {
  pocketkaldi::util::ReadableFile fd_vn;
  std::string fn;
  pocketkaldi::Configuration conf;
  pocketkaldi::Status status_vn;
  status_vn = conf.Read(filename);
  if (!status_vn.ok()) goto pk_load_failed;

  // FST
  fn = conf.GetPathOrElse("fst", "");
  if (fn == "") {
    status_vn = pocketkaldi::Status::Corruption(pocketkaldi::util::Format(
        "Unable to find key 'fst' in {}",
        filename));
    goto pk_load_failed;
  }
  status_vn = fd_vn.Open(fn.c_str());
  self->fst = new pocketkaldi::Fst();
  status_vn = self->fst->Read(&fd_vn);
  if (!status_vn.ok()) goto pk_load_failed;

  // CMVN
  status_vn = ReadCMVN(self, conf);
  if (!status_vn.ok()) goto pk_load_failed;

  // AM
  self->am = new AcousticModel();
  status_vn = self->am->Read(conf);
  if (!status_vn.ok()) goto pk_load_failed;

  // SYMBOL TABLE
  status_vn = ReadSymbolTable(self, conf);
  if (!status_vn.ok()) goto pk_load_failed;

  // DelteLmFst (if available)
  status_vn = ReadDeltaLmFst(self, conf);
  if (!status_vn.ok()) goto pk_load_failed;

  // Initialize fbank feature extractor
  self->fbank = new Fbank();

  if (false) {
pk_load_failed:
    if (!status_vn.ok()) {
      PK_STATUS_IOERROR(status, "%s", status_vn.what().c_str());
    } 
    pk_destroy(self);
  }
}

void pk_utterance_init(pk_utterance_t *utt) {
  utt->internal = new pk_utterance_internal_t;
  utt->hyp = NULL;
  utt->loglikelihood_per_frame = 0.0f;
}

void pk_utterance_destroy(pk_utterance_t *utt) {
  if (utt->internal) {
    delete utt->internal;
    utt->internal = NULL;
  }

  free(utt->hyp);
  utt->hyp = NULL;
  utt->loglikelihood_per_frame = 0.0f;
}

void pk_read_audio(
    pk_utterance_t *utt,
    const char *filename,
    pk_status_t *cs) {
  assert(utt->internal && "pk_read_audio: utterance is not initialized");

  pocketkaldi::Status status = pocketkaldi::Read16kPcm(
      filename, &utt->internal->raw_wave);
  ToRawStatus(status, cs);
}

void pk_process(pk_t *recognizer, pk_utterance_t *utt) {
  assert(utt->hyp == NULL && "utt->hyp == NULL expected");

  // Handle empty utterance
  if (utt->internal->raw_wave.Dim() == 0) {
    utt->hyp = (char *)malloc(sizeof(char));
    *(utt->hyp) = '\0';
    return;
  }

  clock_t t;

  // Extract fbank feats from raw_wave
  t = clock();
  Matrix<float> raw_feats;
  recognizer->fbank->Compute(utt->internal->raw_wave, &raw_feats);
  t = clock() - t;
  fputs(pocketkaldi::util::Format("Fbank: {}ms\n", ((float)t) / CLOCKS_PER_SEC  * 1000).c_str(), stderr);

  // Apply CMVN to raw_wave
  Matrix<float> feats(raw_feats.NumRows(), raw_feats.NumCols());
  if (recognizer->enable_cmvn > 0) {
    t = clock();
    CMVN cmvn(*recognizer->cmvn_global_stats, raw_feats);
    for (int frame = 0; frame < raw_feats.NumRows(); ++frame) {
      SubVector<float> frame_raw = feats.Row(frame);
      cmvn.GetFrame(frame, &frame_raw);
    }
    t = clock() - t;
    fprintf(stderr, "CMVN: %lfms\n", ((float)t) / CLOCKS_PER_SEC  * 1000);
  } else {
    feats.CopyFromMat(raw_feats);
  }


  // Start to decode
  Decoder decoder(recognizer->fst, recognizer->delta_lm_fst);
  Decodable decodable(recognizer->am, 0.1, feats);
  t = clock();
  decodable.Compute();
  t = clock() - t;
  fprintf(stderr, "NNET: %lfms\n", ((float)t) / CLOCKS_PER_SEC  * 1000);
  
  // Decoding
  decoder.Decode(&decodable);
  Decoder::Hypothesis hyp = decoder.BestPath();

  // Get final result
  std::string hyp_text;
  std::vector<int> words = hyp.words();
  std::reverse(words.begin(), words.end());
  if (!hyp.words().empty()) {
    for (int word_id : words) {
      // Append the word into hyp
      const char *word = recognizer->symbol_table->Get(word_id);
      hyp_text += word;
      hyp_text += ' ';
    }

    // Copy hyp to utt->hyp
    utt->hyp = (char *)malloc(sizeof(char) * hyp_text.size());
    pk_strlcpy(utt->hyp, hyp_text.data(), hyp_text.size());
    utt->loglikelihood_per_frame = hyp.weight() / feats.NumRows();
  } else {
    utt->hyp = (char *)malloc(sizeof(char));
    *(utt->hyp) = '\0';
  }
}
