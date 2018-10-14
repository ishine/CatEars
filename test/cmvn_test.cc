// 2017-03-08

#include "cmvn.h"

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include "matrix.h"
#include "fbank.h"

using pocketkaldi::Fbank;
using pocketkaldi::CMVN;
using pocketkaldi::Vector;
using pocketkaldi::Matrix;
using pocketkaldi::Read16kPcm;
using pocketkaldi::Status;
using pocketkaldi::util::ReadableFile;

// Read a matrix from a text file and store them into mat. nrows is the rows of
// mat
std::vector<float> ReadArray(const std::string &filename) {
  FILE *fd = fopen(filename.c_str(), "r");
  assert(fd != NULL);

  float val;
  std::vector<float> matrix_data;
  while (fscanf(fd, "%f", &val) != EOF) {
    matrix_data.push_back(val);
  }
  fclose(fd);
  assert(matrix_data.size() % 40 == 0);

  return matrix_data;
}

void TestOnlineCmvn() {
  std::string global_stats_path = TESTDIR "data/cmvn_stats.bin";
  std::string wav_file = TESTDIR "data/en-us-hello.wav";
  std::vector<float> corr = ReadArray(
      TESTDIR "data/fbankcmvnmat_en-us-hello.wav.txt");

  pk_status_t status;
  pk_status_init(&status);

  // Read audio feats
  Vector<float> pcm_data;

  Status s = Read16kPcm(wav_file.c_str(), &pcm_data);
  assert(s.ok());

  // Computes fbank feature
  Fbank fbank;
  Matrix<float> fbank_feat;
  fbank.Process(pcm_data, &fbank_feat);

  // Read global stats from file
  ReadableFile fd;
  s = fd.Open(global_stats_path);
  assert(s.ok());

  Vector<float> global_stats;
  s = global_stats.Read(&fd);
  assert(s.ok());

  // Initialize cmvn and cmvn instance
  CMVN cmvn(global_stats, fbank_feat);

  // Apply CMVN
  Vector<float> feats(fbank_feat.NumCols());
  for (int i = 0; i < fbank_feat.NumRows(); ++i) {
    cmvn.GetFrame(i, &feats);
    for (int d = 0; d < feats.Dim(); ++d) {
      assert(feats(d) - corr[i * feats.Dim() + d] < 0.0001);
    }
  }
}

int main() {
  TestOnlineCmvn();
  return 0;
}
