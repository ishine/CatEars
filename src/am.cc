// Created at 2017-03-22

#include "am.h"

#include "status.h"
#include "matrix.h"
#include "math.h"

namespace pocketkaldi {

AcousticModel::AcousticModel() :
    left_context_(0),
    right_context_(0),
    num_pdfs_(0),
    chunk_size_(0) {
}

AcousticModel::~AcousticModel() {
  left_context_ = 0;
  right_context_ = 0;
  num_pdfs_ = 0;
}

Status AcousticModel::Read(const Configuration &conf) {
  Status status;

  // Read nnet
  std::string nnet_filename;
  status = conf.GetPath("nnet", &nnet_filename);
  if (!status.ok()) return status;
  pk_status_t c_status;
  pk_status_init(&c_status);

  util::ReadableFile fd;
  PK_CHECK_STATUS(fd.Open(nnet_filename));
  PK_CHECK_STATUS(nnet_.Read(&fd));
  fd.Close();

  // Read prior
  std::string prior_filename;
  PK_CHECK_STATUS(conf.GetPath("prior", &prior_filename));
  PK_CHECK_STATUS(fd.Open(prior_filename));
  PK_CHECK_STATUS(log_prior_.Read(&fd));
  log_prior_.ApplyLog();
  fd.Close();

  // Read left and right context
  PK_CHECK_STATUS(conf.GetInteger("left_context", &left_context_));
  PK_CHECK_STATUS(conf.GetInteger("right_context", &right_context_));
  PK_CHECK_STATUS(conf.GetInteger("chunk_size", &chunk_size_));

  // Read tid2pdf_
  std::string tid2pdf_filename;
  status = conf.GetInteger("num_pdfs", &num_pdfs_);
  if (!status.ok()) return status;
  status = conf.GetPath("tid2pdf", &tid2pdf_filename);
  if (!status.ok()) return status;
  status = fd.Open(tid2pdf_filename);
  if (!status.ok()) return status;
  status = tid2pdf_.Read(&fd);
  if (!status.ok()) return status;

  return Status::OK();
}

int AcousticModel::PrepareChunk(
    const MatrixBase<float> &frames,
    int begin_frame,
    Matrix<float> *chunk) {
  assert(begin_frame < frames.NumRows() && "begin_frame out of boundary");
  int frames_remained = frames.NumRows() - begin_frame;
  int chunk_size = frames_remained > chunk_size_  
      ? chunk_size_
      : frames_remained;
  chunk->Resize(chunk_size + left_context_ + right_context_, frames.NumCols());
  int offset = 0;

  // Left context
  for (int lc = left_context_; lc >= 1; --lc) {
    int tl = begin_frame - lc;
    if (tl < 0) tl = 0;
    chunk->Row(offset).CopyFromVec(frames.Row(tl));
    ++offset;
  }

  // Chunk
  for (int t = begin_frame; t < begin_frame + chunk_size; ++t) {
    assert(t < frames.NumRows() && "t out of boundary");
    chunk->Row(offset).CopyFromVec(frames.Row(t));
    ++offset;
  }

  // Right context
  for (int rc = 0; rc < right_context_; ++rc) {
    int tr = begin_frame + chunk_size + rc;
    if (tr >= frames.NumRows()) tr = frames.NumRows() - 1;
    chunk->Row(offset).CopyFromVec(frames.Row(tr));
    ++offset;
  }

  assert(offset == chunk->NumRows());
  return chunk_size;
}

void AcousticModel::Compute(
    const MatrixBase<float> &frames,
    Matrix<float> *log_prob) {
  Matrix<float> chunk_input, chunk_output;
  log_prob->Resize(frames.NumRows(), log_prior_.Dim());

  // Propagate chunk by chunk
  for (int t = 0; t < frames.NumRows(); t += chunk_size_) {
    int chunk_size = PrepareChunk(frames, t, &chunk_input);
    nnet_.Propagate(chunk_input, &chunk_output);
    assert(chunk_output.NumRows() == chunk_size && "invalid nnet");

    for (int r = t; r < t + chunk_output.NumRows(); ++r) {
      log_prob->Row(r).CopyFromVec(chunk_output.Row(r - t));
    }
  }

  // Compute log-likelihood
  for (int r = 0; r < log_prob->NumRows(); ++r) {
    SubVector<float> row = log_prob->Row(r);
    row.AddVec(-1.0f, log_prior_);
  }
}

}  // namespace pocketkaldi
