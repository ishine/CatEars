// Created at 2017-03-29

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ce_stt.h"
#include "status.h"
#include "util.h"

using pocketkaldi::util::ReadableFile;
using pocketkaldi::util::Split;
using pocketkaldi::Status;

void Fatal(const std::string &msg) {
  printf("error: %s\n", msg.c_str());
  exit(22);
}

void CheckStatus(const Status &status) {
  if (!status.ok()) {
    printf("pasco: %s\n", status.what().c_str());
    exit(1);
  }
}

// Process one utterance and return its hyp
std::string ProcessAudio(ce_stt_t *recognizer, const std::string &filename) {
  ce_wave_format_t wav_fmt;

  FILE *fd = fopen(filename.c_str(), "r");
  if (NULL == fd) Fatal("unable to open: " + filename);
  if (NULL == ce_read_pcm_header(fd, &wav_fmt)) Fatal(ce_stt_last_error());

  ce_utt_t *utt = ce_utt_init(recognizer, &wav_fmt);
  if (NULL == utt) Fatal(ce_stt_last_error());

  char buffer[1024];
  while (!feof(fd)) {
    int bytes_read = fread(buffer, 1, sizeof(buffer), fd);
    if (bytes_read == 0) break;

    ce_stt_process(utt, buffer, bytes_read);
  }

  ce_stt_end_of_stream(utt);
  std::string hyp = utt->hyp;
  ce_utt_destroy(utt);
  fclose(fd);

  return hyp;
}

// Process a list of utterances
void process_scp(ce_stt_t *recognizer, const char *filename) {
  // Read each line in scp file
  ReadableFile fd;
  Status status = fd.Open(filename);
  CheckStatus(status);

  std::string line;
  while (fd.ReadLine(&line, &status) && status.ok()) {
    std::vector<std::string> fields = Split(line, " ");
    if (fields.size() != 2) {
      printf("scp: unexpected line: %s\n", line.c_str());
      exit(22);
    }

    std::string name = fields[0];
    std::string wav_file = fields[1];
    std::string hyp = ProcessAudio(recognizer, wav_file);
    printf("%s %s\n", name.c_str(), hyp.c_str());
  }
  CheckStatus(status);
}

// Print the usage of this program and exit
void print_usage() {
  puts("Usage: pocketkaldi <model-file> <input-file>");
  puts("  Input-file:");
  puts("    *.wav: decode this file.");
  puts("    *.scp: decode audios listed in it.");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 3) print_usage();

  const char *model_file = argv[1];
  const char *input_file = argv[2];
  if (strlen(input_file) < 4) print_usage();

  ce_stt_t *recognizer = ce_stt_init(model_file);
  if (NULL == recognizer) Fatal(ce_stt_last_error());

  const char *suffix = input_file + strlen(input_file) - 4;
  if (strcmp(suffix, ".wav") == 0) {
    std::string hyp = ProcessAudio(recognizer, input_file);
    puts(hyp.c_str());
  } else {
    process_scp(recognizer, input_file);
  }

  ce_stt_destroy(recognizer);
  return 0;
}
