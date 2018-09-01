// Created at 2016-11-08

#ifndef POCKETKALDI_H_
#define POCKETKALDI_H_

#include <stdbool.h>
#include <stdint.h>
#include <memory>

#ifdef __cplusplus
#define POCKETKALDI_EXPORT extern "C"
#else
#define POCKETKALDI_EXPORT
#endif  // __cplusplus

#define PK_STATUS_MSGMAX 256

namespace pocketkaldi {
class Fst;
class AcousticModel;
class Fbank;
template <class T> class Vector;
class SymbolTable;
}  // namespace pocketkaldi

typedef struct pk_status_t {
  bool ok;
  int errcode;
  char message[PK_STATUS_MSGMAX];
} pk_status_t;

typedef struct pk_t {
  pocketkaldi::Fst *fst;
  pocketkaldi::AcousticModel *am;
  pocketkaldi::Fbank *fbank;
  pocketkaldi::Vector<float> *cmvn_global_stats;
  pocketkaldi::SymbolTable *symbol_table;
} pk_t;

// Initialize the status set to success (ok) state
POCKETKALDI_EXPORT
void pk_status_init(pk_status_t *status);

// Internal struct of utterance
typedef struct pk_utterance_internal_t pk_utterance_internal_t;

// Store the wave data, decoding state and hypothesis ... of an utterance
typedef struct pk_utterance_t {
  pk_utterance_internal_t *internal;
  char *hyp;
  float loglikelihood_per_frame;
} pk_utterance_t;

// Initialize the pocketkaldi recognizer (to the initial state)
void pk_init(pk_t *self);

// Load the model for recognizer from file. If any error occured, status->ok
// will be set to false, and the details could be found in status->message
void pk_load(pk_t *self, const char *filename, pk_status_t *status);

// Destroy the recognizer
void pk_destroy(pk_t *self);

// Initialize the utterance
void pk_utterance_init(pk_utterance_t *utt);

// Destroy the utterance
void pk_utterance_destroy(pk_utterance_t *utt);

// Read audio into an utterance. Currently it only supports 16k, mono-channel,
// 16 bit PCM format. On failed, status->ok will be set to false. And the
// datails cound be found in status->message 
void pk_read_audio(
    pk_utterance_t *utt,
    const char *filename,
    pk_status_t *status);

// Process the utterance and set the utt->hyp
void pk_process(pk_t *recognizer, pk_utterance_t *utt);

#endif  // POCKETKALDI_H_

