// pocketkaldi.h -- Created at 2016-11-08
// pasco.h -- Renamed at 2018-10-20

#ifndef PASCO_H_
#define PASCO_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
#define PASCO_EXPORT extern "C"
#else
#define PASCO_EXPORT
#endif  // __cplusplus

#define PASCO_FAILED -1

// Pcm audio format
typedef struct pasco_wave_format_t {
  int num_channels;
  int sample_rate;
  int bits_per_sample;
} pasco_wave_format_t;

// Stores the model for pasco decoder
typedef struct pasco_t pasco_t;

// Internal struct of utterance
typedef struct pasco_utt_internal_t pasco_utt_internal_t;

// Store intermediate data and hypothesis of an utterance in decoding
typedef struct pasco_utt_t {
  pasco_utt_internal_t *internal;
  char *hyp;
  float loglikelihood_per_frame;
} pasco_utt_t;

// Initialize the pasco recognizer (to the initial state)
PASCO_EXPORT
pasco_t *pasco_init(const char *config_file);

// Destroy the recognizer
PASCO_EXPORT
void pasco_destroy(pasco_t *r);

// Initialize and create a new instance of utterance. If error occured, it will
// return NULL and the error could be got by last_error()
PASCO_EXPORT
pasco_utt_t *pasco_utt_init(pasco_t *r, const pasco_wave_format_t *format);

// Destroy the utterance
PASCO_EXPORT
void pasco_utt_destroy(pasco_utt_t *utt);

// Process data from wave stream. it will returns the number of samples read.
// If any error occured, it will return PASCO_FAILED and error message could
// be got by last_error()
PASCO_EXPORT
int32_t pasco_process(pasco_utt_t *utt, const char *data, int32_t size);

// Tell the decoder that the stream is ended.
PASCO_EXPORT
void pasco_end_of_stream(pasco_utt_t *utt);

// Read the hedaer of a .wav file and store the format, then return the pointer
// to format. If error occured during reading, return nullptr and the error
// message could be got by last_error()
PASCO_EXPORT
pasco_wave_format_t *pasco_read_pcm_header(FILE *fd,
                                           pasco_wave_format_t *format);


// Get last error in pasco
PASCO_EXPORT
const char *pasco_last_error();

#endif  // POCKETKALDI_H_

