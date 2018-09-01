// Created at 2017-03-29

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pocketkaldi.h"
#include "status.h"
#include "util.h"

using pocketkaldi::Status;
using pocketkaldi::util::ReadableFile;
using pocketkaldi::util::Split;

void check_and_report_error(const pk_status_t *status) {
  if (!status->ok) {
    printf("pocketkaldi: %s\n", status->message);
    exit(1);
  }
}

void CheckStatus(const Status &status) {
  if (!status.ok()) {
    printf("pocketkaldi: %s\n", status.what().c_str());
    exit(1);
  }
}

// Process one utterance and return its hyp
std::string ProcessAudio(pk_t *recognizer, const std::string &filename) {
  pk_status_t status;
  pk_status_init(&status);

  pk_utterance_t utt;
  pk_utterance_init(&utt);
  pk_read_audio(&utt, filename.c_str(), &status);
  check_and_report_error(&status);

  pk_process(recognizer, &utt);
  std::string hyp = utt.hyp;

  pk_utterance_destroy(&utt);
  return hyp;
}

// Process a list of utterances
void process_scp(pk_t *recognizer, const char *filename) {
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

  pk_t recognizer;
  pk_status_t status;
  pk_status_init(&status);
  pk_init(&recognizer);
  pk_load(&recognizer, model_file, &status);
  check_and_report_error(&status);

  const char *suffix = input_file + strlen(input_file) - 4;
  if (strcmp(suffix, ".wav") == 0) {
    std::string hyp = ProcessAudio(&recognizer, input_file);
    puts(hyp.c_str());
  } else {
    process_scp(&recognizer, input_file);
  }

  pk_destroy(&recognizer);
  return 0;
}
