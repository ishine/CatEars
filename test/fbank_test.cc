// 2017-02-03

#include "fbank.h"

#include <assert.h>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include "pcm_reader.h"
#include "vector.h"
#include "util.h"

using pocketkaldi::Fbank;
using pocketkaldi::Vector;
using pocketkaldi::Matrix;
using pocketkaldi::ReadPcmHeader;
using pocketkaldi::Status;
using pocketkaldi::SubVector;
using pocketkaldi::Read16kPcm;
using pocketkaldi::WaveReader;
using pocketkaldi::util::ReadableFile;

void TestFbank() {
  std::string wav_file = TESTDIR "data/en-us-hello.wav";
  std::string kaldi_featdump = TESTDIR "data/fbankmat_en-us-hello.wav.txt";

  // Read wave file
  Vector<float> pcm_data;

  Status status = Read16kPcm(wav_file.c_str(), &pcm_data);
  puts(status.what().c_str());
  assert(status.ok());

  // Calculate fbank feature
  Fbank fbank;
  Fbank::Instance fbank_inst;
  Matrix<float> fbank_feat;

  fbank.Process(&fbank_inst, pcm_data, &fbank_feat);
  std::vector<float> fbank_featvec;
  for (int i = 0; i < fbank_feat.NumRows(); ++i) {
    SubVector<float> row = fbank_feat.Row(i);
    for (int j = 0; j < row.Dim(); ++j) {
      fbank_featvec.push_back(row(j));
    }
  }

  // Check the fbank_feat
  std::ifstream is(kaldi_featdump);
  assert(is.is_open());

  float val;
  int line_count = 0;
  for (int i = 0; is >> val; ++i) {
    assert(abs(val - fbank_featvec[i]) < 1e-4);
    ++line_count;
  }
  assert(line_count == 1880);
}

// Split the data into chunks, size for each chunk is chunk_size
std::vector<std::vector<char>> ChunkArray(
    const std::vector<char> &data,
    int chunk_size) {
  int n_chunks = data.size() / chunk_size;
  int last_chunk_size = data.size() % chunk_size;
  if (last_chunk_size == 0) {
    last_chunk_size = chunk_size;
  } else {
    ++n_chunks;
  }

  std::vector<std::vector<char>> chunks;
  for (int i = 0; i < n_chunks; ++i) {
    int cnt_chunk_size = chunk_size;
    if (i == n_chunks - 1) cnt_chunk_size = last_chunk_size;
    chunks.emplace_back(data.data() + i * chunk_size,
                        data.data() + i * chunk_size + cnt_chunk_size);
  }

  return chunks;
}

void TestFbankStreaming() {
  std::string wav_file = TESTDIR "data/en-us-hello.wav";
  std::string kaldi_featdump = TESTDIR "data/fbankmat_en-us-hello.wav.txt";

  // Read wave file
  Vector<float> pcm_data;

  ReadableFile fd_wav;
  Status status = fd_wav.Open(wav_file);
  assert(status.ok());

  ce_wave_format_t fmt;
  status = ReadPcmHeader(&fd_wav, &fmt);
  assert(status.ok());

  std::vector<char> buffer(fd_wav.file_size() - 44);
  status = fd_wav.Read(buffer.data(), buffer.size());
  assert(status.ok());

  // Calculate fbank feature
  Fbank fbank;
  Fbank::Instance fbank_inst;
  Matrix<float> fbank_feat;
  WaveReader wave_reader;
  status = wave_reader.SetFormat(fmt);
  assert(status.ok());

  std::vector<std::vector<char>> chunks = ChunkArray(buffer, 1024);  
  std::vector<float> fbank_featvec;
  for (const std::vector<char> &chunk : chunks) {
    wave_reader.Process(chunk.data(), chunk.size(), &pcm_data);
    fbank.Process(&fbank_inst, pcm_data, &fbank_feat);
    for (int i = 0; i < fbank_feat.NumRows(); ++i) {
      SubVector<float> row = fbank_feat.Row(i);
      for (int j = 0; j < row.Dim(); ++j) {
        fbank_featvec.push_back(row(j));
      }
    }
  }

  // Check the fbank_feat
  std::ifstream is(kaldi_featdump);
  assert(is.is_open());

  float val;
  int line_count = 0;
  for (int i = 0; is >> val; ++i) {
    assert(fabs(val - fbank_featvec[i]) < 1e-4);
    ++line_count;
  }
  assert(line_count == 1880);
}

int main() {
  TestFbank();
  TestFbankStreaming();
  return 0;
}