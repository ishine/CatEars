// Created at 2017-03-13

#ifndef POCKETKALDI_NNET_H_
#define POCKETKALDI_NNET_H_

#include <vector>
#include "matrix.h"
#include "util.h"
#include "gemm.h"

#define PK_NNET_SECTION "NN02"
#define PK_NNET_LAYER_SECTION "LAY0"


namespace pocketkaldi {

// The base class for different type of layers
class Layer {
 public:
  // Kinds of linear types
  enum {
    kLinear = 0,
    kReLU = 1,
    kNormalize = 2,
    kSoftmax = 3,
    kSplice = 6,
    kBatchNorm = 7,
    kLogSoftmax = 8,
    kNarrow = 9
  };

  // Propogate a batch of input vectors through this layer. And the batch of
  // output vectors are in `out`
  virtual void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const = 0;

  // Read layer from fd
  virtual Status Read(util::ReadableFile *fd) = 0;

  // Layer type
  virtual std::string Type() const = 0;

  virtual ~Layer() {}
};

// Linear layer: x^T dot W + b
class LinearLayer : public Layer {
 public:
  LinearLayer();
  // Initialize the linear layer with parameter W and b. It just copies the
  // values from W and b.
  LinearLayer(
      const MatrixBase<float> &W,
      const VectorBase<float> &b);

  // Implements interface Layer
  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override;

  // Implements interface Layer
  std::string Type() const override { return "Linear"; }

 private:
  Matrix<float> W_;
  Vector<float> b_;
};

// SpliceLayer splices input matrix with each indcies. For example
// Input matrix is [v1, v2, v3, v4]
// indcies: -2, 0, 1
// Output matrxi is:
//    [[concat(v1, v1, v2)],
//     [concat(v1, v2, v3)],
//     [concat(v1, v3, v4)],
//     [concat(v2, v4, v4)]]
class SpliceLayer : public Layer {
 public:
  SpliceLayer();
  SpliceLayer(const std::vector<int> &indices);

  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override;

  // Implements interface Layer
  std::string Type() const override { return "Splice"; }

 private:
  std::vector<int> indices_;
};

// BatchNormLayer is a layer to apply batch normalization without affine,
// conputation is:
//   y = (x - E(x)) / sqrt(VAR(x) + eps) 
class BatchNormLayer : public Layer {
 public:
  BatchNormLayer();
  BatchNormLayer(float eps);

  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;
 
  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override;

  // Implements interface Layer
  std::string Type() const override { return "BatchNorm"; }

 private:
  float eps_;
};

// Softmax layer
class SoftmaxLayer : public Layer {
 public:
  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override {};

  // Implements interface Layer
  std::string Type() const override { return "Softmax"; }
};

// LogSoftMax layer 
class LogSoftmaxLayer : public Layer {
 public:
  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override {};

  // Implements interface Layer
  std::string Type() const override { return "LogSoftmax"; }
};

// ReLU layer
class ReLULayer : public Layer {
 public:
  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override {};

  // Implements interface Layer
  std::string Type() const override { return "ReLU"; }
};

// Normalize layer
class NormalizeLayer : public Layer {
 public:
  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override {};

  // Implements interface Layer
  std::string Type() const override { return "Normalize"; }
};

class NarrowLayer : public Layer {
 public:
  NarrowLayer();
  NarrowLayer(int narrow_left, int narrow_right);

  void Propagate(
      const MatrixBase<float> &in,
      Matrix<float> *out) const override;

  // Implements interface Layer
  Status Read(util::ReadableFile *fd) override;

  // Implements interface Layer
  std::string Type() const override { return "NarrowLayer"; }

 private:
  int narrow_left_;
  int narrow_right_;
};

// The neural network class. It have a stack of different kinds of `Layer`
// instances. And the batch matrix could be propogate through this neural
// network using `Propagate` method
class Nnet {
 public:
  Nnet();

  // Read the nnet from file
  Status Read(util::ReadableFile *fd);

  // Propogate batch matrix through this neural network
  void Propagate(const MatrixBase<float> &in, Matrix<float> *out) const;

  // Returns the left and right context of nnet
  int left_context() const { return left_context_; }
  int right_context() const { return right_context_; }

 private:
  std::vector<std::unique_ptr<Layer>> layers_;

  // Read a layer from `fd` and store into layers_
  Status ReadLayer(util::ReadableFile *fd);

  int left_context_;
  int right_context_;
};

}  // namespace pocketkaldi

#endif
