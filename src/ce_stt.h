// pocketkaldi.h -- Created at 2016-11-08
// pasco.h -- Renamed at 2018-10-20
// ce_stt.h -- Renamed at 2019-02-08

#ifndef CE_STT_H_
#define CE_STT_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
#define CE_STT_EXPORT extern "C"
#else
#define CE_STT_EXPORT
#endif  // __cplusplus

#define CE_STT_FAILED -1

// Pcm audio format
typedef struct ce_wave_format_t {
  int num_channels;
  int sample_rate;
  int bits_per_sample;
} ce_wave_format_t;

// Stores the model for pasco decoder
typedef struct ce_stt_t ce_stt_t;

// Internal struct of utterance
typedef struct ce_utt_internal_t ce_utt_internal_t;

// Store intermediate data and hypothesis of an utterance in decoding
typedef struct ce_utt_t {
  ce_utt_internal_t *internal;
  char *hyp;
  float loglikelihood_per_frame;
} ce_utt_t;

// Initialize the pasco recognizer (to the initial state)
CE_STT_EXPORT
ce_stt_t *ce_stt_init(const char *config_file);

// Destroy the recognizer
CE_STT_EXPORT
void ce_stt_destroy(ce_stt_t *r);

// Initialize and create a new instance of utterance. If error occured, it will
// return NULL and the error could be got by last_error()
CE_STT_EXPORT
ce_utt_t *ce_utt_init(ce_stt_t *r, const ce_wave_format_t *format);

// Destroy the utterance
CE_STT_EXPORT
void ce_utt_destroy(ce_utt_t *utt);

// Process data from wave stream. it will returns the number of samples read.
// If any error occured, it will return PASCO_FAILED and error message could
// be got by last_error()
CE_STT_EXPORT
int32_t ce_stt_process(ce_utt_t *utt, const char *data, int32_t size);

// Tell the decoder that the stream is ended.
CE_STT_EXPORT
void ce_stt_end_of_stream(ce_utt_t *utt);

// Read the hedaer of a .wav file and store the format, then return the pointer
// to format. If error occured during reading, return nullptr and the error
// message could be got by last_error()
CE_STT_EXPORT
ce_wave_format_t *ce_read_pcm_header(FILE *fd, ce_wave_format_t *format);


// Get last error in pasco
CE_STT_EXPORT
const char *ce_stt_last_error();

#endif  // POCKETKALDI_H_

