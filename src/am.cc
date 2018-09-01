// Created at 2017-03-22

#include "am.h"

#include "status.h"
#include "matrix.h"
#include "math.h"

namespace pocketkaldi {

AcousticModel::AcousticModel() :
    left_context_(0),
    right_context_(0),
    num_pdfs_(0) {
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
  status = conf.GetInteger("left_context", &left_context_);
  status = conf.GetInteger("right_context", &right_context_);
  if (!status.ok()) return status;

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

void AcousticModel::Compute(
    const MatrixBase<float> &frames,
    Matrix<float> *log_prob) {
  // Propogate through the neural network
  nnet_.Propagate(frames, log_prob);

  // Compute log-likelihood
  for (int r = 0; r < log_prob->NumRows(); ++r) {
    SubVector<float> row = log_prob->Row(r);
    row.AddVec(-1.0f, log_prior_);
  }
}

}  // namespace pocketkaldi
