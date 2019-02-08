// Created at 2017-03-29

#include "ce_stt.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <algorithm>
#include <string>
#include "am.h"
#include "cmvn.h"
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
using pocketkaldi::Status;
using pocketkaldi::Configuration;
using pocketkaldi::Vector;
using pocketkaldi::VectorBase;
using pocketkaldi::Matrix;
using pocketkaldi::SubVector;
using pocketkaldi::AcousticModel;
using pocketkaldi::SymbolTable;
using pocketkaldi::LmFst;
using pocketkaldi::DeltaLmFst;
using pocketkaldi::util::Format;
using pocketkaldi::util::ReadableFile;
using pocketkaldi::ReadPcmHeader;
using pocketkaldi::WaveReader;


typedef struct ce_stt_t {
  pocketkaldi::Fst *fst;
  pocketkaldi::LmFst *large_lm_fst;
  pocketkaldi::DeltaLmFst *delta_lm_fst;
  pocketkaldi::Vector<float> *original_lm;
  pocketkaldi::AcousticModel *am;
  pocketkaldi::Fbank *fbank;
  pocketkaldi::SymbolTable *symbol_table;
} ce_stt_t;

// The internal version of an utterance. It stores the intermediate state in
// decoding.
typedef struct ce_utt_internal_t {
  const ce_stt_t *recognizer;

  WaveReader wave_reader;
  Fbank::Instance fbank_inst;
  AcousticModel::Instance am_inst;
  std::unique_ptr<Decoder> decoder;
} ce_utt_internal_t;

namespace {

// Buffer for last_error()
char error_message[2048] = "";

// Read symbol table
Status ReadSymbolTable(ce_stt_t *self, const Configuration &conf) {
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
Status ReadDeltaLmFst(ce_stt_t *self, const Configuration &conf) {
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
  self->large_lm_fst->InitBucket0();
  
  // Build DeltaLmFst
  assert(self->symbol_table != nullptr);
  self->delta_lm_fst = new DeltaLmFst(self->original_lm,
                                      self->large_lm_fst,
                                      self->symbol_table);

  return Status::OK();
}

// Reads the HCLG fst
Status ReadHclgFst(ce_stt_t *self, const Configuration &conf) {
  ReadableFile fd;
  std::string filename = conf.GetPathOrElse("fst", "");
  if (filename == "") {
    return Status::Corruption(
        Format("Unable to find key 'fst' in {}", filename));
  }
  PK_CHECK_STATUS(fd.Open(filename.c_str()));
  self->fst = new pocketkaldi::Fst();
  PK_CHECK_STATUS(self->fst->Read(&fd));

  return Status::OK();
}

// Checks if parameter utt is correct. On success return 0, on failed return
// CE_STT_FAILED and copy error string into error_message
int32_t CheckParamUtt(ce_utt_t *utt) {
  if (utt == nullptr) {
    pasco_strlcpy(error_message, "utt is NULL", sizeof(error_message));
    return CE_STT_FAILED;
  }
  if (utt->internal == nullptr) {
    pasco_strlcpy(error_message, "utt->internal is NULL", sizeof(error_message));
    return CE_STT_FAILED;
  }

  const ce_stt_t *recognizer = utt->internal->recognizer;
  if (recognizer == nullptr) {
    pasco_strlcpy(error_message,
                  "utt->internal->recognizer is NULL", 
                  sizeof(error_message));
    return CE_STT_FAILED;
  }

  return 0;
}

// Get the hypothesis from best path in pattice and convert it into text format.
// Then store into utt->hyp
void StoreHypText(ce_utt_t *utt) {
  PK_DEBUG("StoreHypText()");

  // Decoding
  const ce_stt_t *recognizer = utt->internal->recognizer;
  Decoder *decoder = utt->internal->decoder.get();
  Decoder::Hypothesis hyp = decoder->BestPath();

  // Get final result
  std::string text;
  std::vector<int> words = hyp.words();
  std::reverse(words.begin(), words.end());
  if (!hyp.words().empty()) {
    for (int word : words) {
      // Append the word into hyp
      text += recognizer->symbol_table->Get(word);
      text += ' ';
    }

    // Copy hyp to utt->hyp
    delete[] utt->hyp;
    utt->hyp = new char[text.size()];

    // pasco_strlcpy will fill the last space as '\0' automatically
    pasco_strlcpy(utt->hyp, text.data(), text.size());
    utt->loglikelihood_per_frame = hyp.weight() / decoder->NumFramesDecoded();
  } else {
    delete[] utt->hyp;
    utt->hyp = new char[1];
    *(utt->hyp) = '\0';
  }
}

}  // namespace

ce_stt_t *ce_stt_init(const char *config_file) {
  ce_stt_t *recognizer = new ce_stt_t;
  memset(recognizer, '\0', sizeof(ce_stt_t));

  Status status;

  Configuration conf;
  status = conf.Read(config_file);
  if (!status.ok()) goto pasco_init_failed;

  // FST
  status = ReadHclgFst(recognizer, conf);
  if (!status.ok()) goto pasco_init_failed;

  // AM
  recognizer->am = new AcousticModel();
  status = recognizer->am->Read(conf);
  if (!status.ok()) goto pasco_init_failed;

  // SYMBOL TABLE
  status = ReadSymbolTable(recognizer, conf);
  if (!status.ok()) goto pasco_init_failed;

  // DelteLmFst (if available)
  status = ReadDeltaLmFst(recognizer, conf);
  if (!status.ok()) goto pasco_init_failed;

  // Initialize fbank feature extractor
  recognizer->fbank = new Fbank();

  return recognizer;

  if (false) {
pasco_init_failed:
    pasco_strlcpy(error_message, status.what().c_str(), sizeof(error_message));
    ce_stt_destroy(recognizer);
    return nullptr;
  }
}

void ce_stt_destroy(ce_stt_t *recognizer) {
  delete recognizer->fst;
  recognizer->fst = nullptr;

  delete recognizer->large_lm_fst;
  recognizer->large_lm_fst = nullptr;

  delete recognizer->delta_lm_fst;
  recognizer->delta_lm_fst = nullptr;

  delete recognizer->original_lm;
  recognizer->original_lm = nullptr;

  delete recognizer->am;
  recognizer->am = NULL;

  delete recognizer->symbol_table;
  recognizer->symbol_table = nullptr;

  delete recognizer->fbank;
  recognizer->fbank = NULL;
}

ce_utt_t *ce_utt_init(ce_stt_t *recognizer, const ce_wave_format_t *format) {
  ce_utt_t *c_utt = new ce_utt_t;
  ce_utt_internal_t *utt = new ce_utt_internal_t;

  utt->decoder = std::unique_ptr<Decoder>(new Decoder(
      recognizer->fst,
      recognizer->am->TransitionPdfIdMap(),
      0.1,
      recognizer->delta_lm_fst));
  utt->decoder->Initialize();

  c_utt->hyp = new char[1];
  *(c_utt->hyp) = '\0';
  c_utt->loglikelihood_per_frame = 0.0f;
  c_utt->internal = utt;

  // Set wave format in wave reader
  Status status = utt->wave_reader.SetFormat(*format);
  if (!status.ok()) {
    pasco_strlcpy(error_message, status.what().c_str(), sizeof(error_message));
    ce_utt_destroy(c_utt);
    return nullptr;
  }

  utt->recognizer = recognizer;

  return c_utt;
}

void ce_utt_destroy(ce_utt_t *c_utt) {
  delete[] c_utt->hyp;
  c_utt->hyp = nullptr;

  c_utt->loglikelihood_per_frame = 0.0f;

  delete c_utt->internal;
  delete c_utt;
}

int32_t ce_stt_process(ce_utt_t *c_utt, const char *data, int32_t size) {
  Vector<float> samples;
  Matrix<float> feats;
  Matrix<float> log_prob;

  if (CE_STT_FAILED == CheckParamUtt(c_utt)) {
    return CE_STT_FAILED;
  }
  const ce_stt_t *recognizer = c_utt->internal->recognizer;
  ce_utt_internal_t *utt = c_utt->internal;

  // Bytes to samples
  Status status = utt->wave_reader.Process(data, size, &samples);
  PK_DEBUG(Format("{} samples read", samples.Dim()));
  if (!status.ok()) goto pasco_process_failed;
  if (samples.Dim() == 0) return 0;

  // Samples to fbank features
  recognizer->fbank->Process(&utt->fbank_inst, samples, &feats);
  PK_DEBUG(Format("get {} frames of fbank feature", feats.NumRows()));

  // Compute log_prob by AM
  for (int frame_idx = 0; frame_idx < feats.NumRows(); ++frame_idx) {
    SubVector<float> frame_feats = feats.Row(frame_idx);
    recognizer->am->Process(&utt->am_inst, frame_feats, &log_prob);
    if (log_prob.NumRows() != 0) {
      PK_DEBUG(Format("get {} frames of log_prob", log_prob.NumRows()));
      for (int r = 0; r < log_prob.NumRows(); ++r) {
        utt->decoder->Process(log_prob.Row(r));

        // Update hypothesis
        if (utt->decoder->NumFramesDecoded() % 20 == 0) {
          StoreHypText(c_utt);
        }
      }
    }
  }

  return samples.Dim();

  if (false) {
pasco_process_failed:
    pasco_strlcpy(error_message, status.what().c_str(), sizeof(error_message));
    return CE_STT_FAILED;
  }
}

void ce_stt_end_of_stream(ce_utt_t *c_utt) {
  PK_DEBUG("pasco_end_of_stream()");
  ce_utt_internal_t *utt = c_utt->internal;

  if (CE_STT_FAILED == CheckParamUtt(c_utt)) {
    return;
  }
  const ce_stt_t *recognizer = c_utt->internal->recognizer;

  // Process remained frames in AM
  Matrix<float> log_prob;
  recognizer->am->EndOfStream(&utt->am_inst, &log_prob);
  if (log_prob.NumRows() != 0) {
    for (int r = 0; r < log_prob.NumRows(); ++r) {
      utt->decoder->Process(log_prob.Row(r));
    }
  }
  utt->decoder->EndOfStream();

  StoreHypText(c_utt);
}

ce_wave_format_t *ce_read_pcm_header(FILE *fp, ce_wave_format_t *format) {
  ReadableFile fd(fp);
  Status status = ReadPcmHeader(&fd, format);
  if (!status.ok()) {
    pasco_strlcpy(error_message, status.what().c_str(), sizeof(error_message));
    return nullptr;
  }

  return format;
}

const char *ce_stt_last_error() {
  return error_message;
}
