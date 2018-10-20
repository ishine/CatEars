// Created at 2017-03-22

#include "am.h"

#include "status.h"
#include "matrix.h"
#include "math.h"

namespace pocketkaldi {

AcousticModel::Instance::Instance(): started(false) {}

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

void AcousticModel::AppendFrame(Instance *inst,
                                const VectorBase<float> &frame_feat) const {
  Vector<float> frame(frame_feat.Dim());
  frame.CopyFromVec(frame_feat);
  inst->feats_buffer.emplace_back(std::move(frame));
}

bool AcousticModel::BatchAvailable(Instance *inst) const {
  int frames_available = inst->feats_buffer.size();
  if (frames_available >= left_context_ + right_context_ + chunk_size_) {
    return true;
  } else {
    return false;
  }
}

void AcousticModel::ComputeBatch(Instance *inst,
                                 int batch_size,
                                 Matrix<float> *log_prob) const {
  if (batch_size == kBatchSizeAll) {
    batch_size = inst->feats_buffer.size() - left_context_ - right_context_;
    assert(batch_size > 0 && "ComputeBatch: insufficient data");
  }
  if (batch_size == 0) {
    log_prob->Resize(0, log_prob->NumCols());
    return;
  }

  // Prepare input matrix
  int batch_input_size = batch_size + left_context_ + right_context_;
  assert(inst->feats_buffer.size() >= batch_input_size &&
         "ComputeBatch: insufficient data");
  int feat_dim = inst->feats_buffer[0].Dim();
  Matrix<float> batch_input(batch_input_size, feat_dim);
  for (int i = 0; i < batch_input_size; ++i) {
    batch_input.Row(i).CopyFromVec(inst->feats_buffer[i]);
  }

  // Propogate through nn
  nnet_.Propagate(batch_input, log_prob);
  assert(log_prob->NumRows() == batch_size && "invalid nnet");

  // Compute log-likelihood
  for (int r = 0; r < log_prob->NumRows(); ++r) {
    SubVector<float> row = log_prob->Row(r);
    row.AddVec(-1.0f, log_prior_);
  }
}

void AcousticModel::Process(Instance *inst,
                            const VectorBase<float> &frame_feat,
                            Matrix<float> *log_prob) const {
  // Add left padding frames
  if (!inst->started) {
    for (int i = 0; i < left_context_; ++i) {
      AppendFrame(inst, frame_feat);
    }
    inst->started = true;
  }

  // Add current frame
  AppendFrame(inst, frame_feat);

  
  if (!BatchAvailable(inst)) {
    log_prob->Resize(0, 0);
    return;
  }

  // A batch of frames is available
  ComputeBatch(inst, chunk_size_, log_prob);

  // Remove useless frames
  for (int i = 0; i < chunk_size_; ++i) {
    inst->feats_buffer.pop_front();
  }
}

void AcousticModel::EndOfStream(Instance *inst, Matrix<float> *log_prob) const {
  // Do nothing if feature buffer is empty
  if (inst->feats_buffer.empty()) {
    log_prob->Resize(0, 0);
    return;
  }

  // Add right padding frames
  const VectorBase<float> &last_frame = inst->feats_buffer.back();
  for (int i = 0; i < right_context_; ++i) {
    AppendFrame(inst, last_frame);
  }

  // Do nothing when no enough frames to compute
  if (inst->feats_buffer.size() <= left_context_ + right_context_) {
    log_prob->Resize(0, 0);
    return;
  }

  ComputeBatch(inst, kBatchSizeAll, log_prob);
}

}  // namespace pocketkaldi
