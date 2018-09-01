// Created at 2017-03-13

#include "nnet.h"

#include <assert.h>
#include <math.h>
#include "gemm.h"

namespace pocketkaldi {

LinearLayer::LinearLayer() {}
LinearLayer::LinearLayer(
    const MatrixBase<float> &W,
    const VectorBase<float> &b) {
  assert(b.Dim() == W.NumRows() && 
         "linear layer: dimension mismatch in W and b");
  W_.Resize(W.NumCols(), W.NumRows());
  W_.CopyFromMat(W, MatrixBase<float>::kTrans);
  b_.Resize(b.Dim());
  b_.CopyFromVec(b);
}

void LinearLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  assert(b_.Dim() != 0 && "LinearLayer is not initialized");
  out->Resize(in.NumRows(), W_.NumCols());

  // xW
  GEMM<float> sgemm;
  MatMat(in, W_, out, &sgemm);

  // + b
  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> row = out->Row(row_idx);
    row.AddVec(1.0f, b_);
  }
}

Status LinearLayer::Read(util::ReadableFile *fd) {
  PK_CHECK_STATUS(W_.Read(fd));
  PK_CHECK_STATUS(b_.Read(fd));

  return Status::OK();
}

SpliceLayer::SpliceLayer() {}
SpliceLayer::SpliceLayer(const std::vector<int> &indices):
    indices_(indices) {
}

void SpliceLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  assert(indices_.size() != 0 && "SpliceLayer is not initialized");
  if (in.NumRows() == 0 || in.NumCols() == 0) return;

  int out_cols = indices_.size() * in.NumCols();
  out->Resize(in.NumRows(), out_cols);

  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> out_row = out->Row(row_idx);
    int offset = 0;

    // Left context
    for (int c : indices_) {
      int cnt_idx = row_idx + c;
      if (cnt_idx < 0) cnt_idx = 0;
      if (cnt_idx > in.NumRows() - 1) cnt_idx = in.NumRows() - 1;
      SubVector<float> v = out_row.Range(offset, in.NumCols());
      v.CopyFromVec(in.Row(cnt_idx));
      offset += in.NumCols();
    }

    assert(offset == out_cols && "splice: offset and size mismatch");
  }
}

Status SpliceLayer::Read(util::ReadableFile *fd) {
  // Clean indices_
  indices_.clear();

  int32_t num_indcies = 0;
  PK_CHECK_STATUS(fd->ReadValue<int32_t>(&num_indcies));
  if (num_indcies < 0) {
    return Status::Corruption("SpliceLayer: unexpected num_indcies");
  }

  // Read each index
  int32_t index = 0;
  for (int i = 0; i < num_indcies; ++i) {
    PK_CHECK_STATUS(fd->ReadValue<int32_t>(&index));
    indices_.push_back(index);
  }

  return Status::OK();
}

BatchNormLayer::BatchNormLayer(): eps_(-1) {}
BatchNormLayer::BatchNormLayer(float eps): eps_(eps) {}

void BatchNormLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  assert(eps_ >= 0 && "BatchNormLayer is not initialized");
  Vector<float> mean(in.NumCols(), Vector<float>::kSetZero);
  Vector<float> scale(in.NumCols(), Vector<float>::kSetZero);

  for (int r = 0; r < in.NumRows(); ++r) {
    for (int c = 0; c < in.NumCols(); ++c) {
      mean(c) += in(r, c);
      scale(c) += in(r, c) * in(r, c);
    }
  }

  mean.Scale(1.0f / in.NumRows());

  Vector<float> mean2(mean.Dim());
  mean2.CopyFromVec(mean);
  mean2.ApplyPow(2);

  // Currently scale is VAR(x)
  scale.Scale(1.0f / in.NumRows());
  scale.AddVec(-1.0, mean2);

  // Scale = 1 / sqrt(VAR(x) + eps)
  scale.Add(eps_);
  scale.ApplyFloor(1e-5);
  scale.ApplyPow(-0.5f);

  // Write to out
  out->Resize(in.NumRows(), in.NumCols());
  for (int r = 0; r < in.NumRows(); ++r) {
    SubVector<float> row = out->Row(r);
    row.CopyFromVec(in.Row(r));
    row.AddVec(-1.0f, mean);
    row.MulElements(scale);
  }
}

Status BatchNormLayer::Read(util::ReadableFile *fd) {
  float eps = 0;
  PK_CHECK_STATUS(fd->ReadValue<float>(&eps));
  eps_ = eps;
}

void SoftmaxLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  out->CopyFromMat(in);
  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> row = out->Row(row_idx);
    row.ApplySoftMax();
  }
}

void LogSoftmaxLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  out->CopyFromMat(in);
  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> row = out->Row(row_idx);
    row.ApplyLogSoftMax();
  }
}


void ReLULayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  out->CopyFromMat(in);
  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> row = out->Row(row_idx);
    for (int col_idx = 0; col_idx < row.Dim(); ++ col_idx) {
      if (row(col_idx) < 0.0f) row(col_idx) = 0.0f;
    }
  }
}

void NormalizeLayer::Propagate(
    const MatrixBase<float> &in,
    Matrix<float> *out) const {
  float D = in.NumCols();
  out->Resize(in.NumRows(), in.NumCols());
  out->CopyFromMat(in);
  for (int row_idx = 0; row_idx < out->NumRows(); ++row_idx) {
    SubVector<float> row = out->Row(row_idx);

    double squared_sum = row.VecVec(row);
    float scale = static_cast<float>(sqrt(D / squared_sum));
    row.Scale(scale);
  }
}

Nnet::Nnet() {
}

Status Nnet::ReadLayer(util::ReadableFile *fd) {
  Matrix<float> W;
  Vector<float> b;

  // Read section name
  PK_CHECK_STATUS(fd->ReadAndVerifyString(PK_NNET_LAYER_SECTION));

  // Read layer type
  int layer_type;
  PK_CHECK_STATUS(fd->ReadValue<int32_t>(&layer_type));

  // Read additional parameters and initialize layer
  std::unique_ptr<Layer> layer = nullptr;
  switch (layer_type) {
  case Layer::kLinear:
    layer = std::unique_ptr<Layer>(new LinearLayer());
    break;
  case Layer::kReLU:
    layer = std::unique_ptr<Layer>(new ReLULayer());
    break;
  case Layer::kNormalize:
    layer = std::unique_ptr<Layer>(new NormalizeLayer());
    break;
  case Layer::kSoftmax:
    layer = std::unique_ptr<Layer>(new SoftmaxLayer());
    break;
  case Layer::kSplice:
    layer = std::unique_ptr<Layer>(new SpliceLayer());
    break;
  case Layer::kBatchNorm:
    layer = std::unique_ptr<Layer>(new BatchNormLayer());
    break;
  case Layer::kLogSoftmax:
    layer = std::unique_ptr<Layer>(new LogSoftmaxLayer());
    break;
  default:
    return Status::Corruption(util::Format(
        "read_layer: unexpected layer type: {} ({})",
        layer_type,
        fd->filename()));
  }

  // Read the content of this layer (if have)
  PK_CHECK_STATUS(layer->Read(fd));
  layers_.emplace_back(std::move(layer));

  return Status::OK();
}

Status Nnet::Read(util::ReadableFile *fd) {
  // Read section name
  PK_CHECK_STATUS(fd->ReadAndVerifyString(PK_NNET_SECTION));

  int num_layers;
  PK_CHECK_STATUS(fd->ReadValue<int32_t>(&num_layers));

  // Read each layers
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    PK_CHECK_STATUS(ReadLayer(fd));
  }

  return Status::OK();
}

void Nnet::Propagate(const MatrixBase<float> &in, Matrix<float> *out) const {
  Matrix<float> layer_input, layer_output;
  
  layer_input.Resize(in.NumRows(), in.NumCols());
  layer_input.CopyFromMat(in);
  for (const std::unique_ptr<Layer> &layer : layers_) {
    layer->Propagate(layer_input, &layer_output);
    layer_input.Swap(&layer_output);
  }

  out->Resize(layer_input.NumRows(), layer_input.NumCols());
  out->CopyFromMat(layer_input);
}

}  // namespace pocketkaldi
