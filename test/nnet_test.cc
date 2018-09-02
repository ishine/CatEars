// Created at 2017-03-13

#include "nnet.h"

#include <assert.h>
#include <math.h>
#include "matrix.h"

using pocketkaldi::Layer;
using pocketkaldi::LinearLayer;
using pocketkaldi::SoftmaxLayer;
using pocketkaldi::ReLULayer;
using pocketkaldi::NormalizeLayer;
using pocketkaldi::BatchNormLayer;
using pocketkaldi::LogSoftmaxLayer;
using pocketkaldi::SpliceLayer;
using pocketkaldi::Matrix;
using pocketkaldi::SubMatrix;
using pocketkaldi::Vector;
using pocketkaldi::SubVector;
using pocketkaldi::NarrowLayer;

bool CheckEq(float a, float b) {
  return fabs(a - b) < 1e-3;
}

// Checks if v has the same data as std:;vector ref
bool CheckVector(const SubVector<float> &v, std::vector<float> ref) {
  if (v.Dim() != static_cast<int>(ref.size())) return false;
  for (int i = 0; i < v.Dim(); ++i) {
    if (CheckEq(v(i), ref[i]) == false) return false;
  }

  return true;
}

void TestSpliceLayer() {
  float x_data[] = {
    1, 1,
    2, 2,
    3, 3,
    4, 4
  };

  SubMatrix<float> x(x_data, 4, 2, 2);
  SpliceLayer spliceLayer({-2, 1});

  Matrix<float> y;
  spliceLayer.Propagate(x, &y);

  // Check results
  assert(y.NumCols() == 4 && y.NumRows() == 4);
  assert(CheckVector(y.Row(0), {1, 1, 2, 2}));
  assert(CheckVector(y.Row(1), {1, 1, 3, 3}));
  assert(CheckVector(y.Row(2), {1, 1, 4, 4}));
  assert(CheckVector(y.Row(3), {2, 2, 4, 4}));
}

void TestLinearLayer() {
  // Matrix W
  float W_data[] = {
    0.1, 0.8, 0.9,
    0.4, 0.2, 0.7,
    0.2, 0.1, 0.1,
    0.4, 0.3, 0.2
  };
  SubMatrix<float> W(W_data, 4, 3, 3);

  // Vector b
  float b_data[] = {0.1, -0.1, 0.2, -0.2};
  SubVector<float> b(b_data, 4);

  // Create the linear layer
  LinearLayer linear(W, b);

  // Propagation
  // Vector x
  //   0.3 -0.1 0.9
  float x_data[] = {0.3, -0.1, 0.9};
  SubMatrix<float> x(x_data, 1, 3, 3);
  Matrix<float> y;
  linear.Propagate(x, &y);

  // Check results
  assert(y.NumCols() == 4 && y.NumRows() == 1);
  assert(CheckEq(y(0, 0), 0.86f));
  assert(CheckEq(y(0, 1), 0.63f));
  assert(CheckEq(y(0, 2), 0.34f));
  assert(CheckEq(y(0, 3), 0.07f));
}


void TestSoftmaxLayer() {
  // Create the Softmax layer
  SoftmaxLayer softmax;

  float x_data[] = {0.3, -0.1, 0.9, 0.2};
  SubMatrix<float> x(x_data, 1, 4, 4);
  Matrix<float> y;
  softmax.Propagate(x, &y);

  // Check results
  assert(y.NumCols() == 4);
  assert(y.NumRows() == 1);
  assert(CheckEq(y(0, 0), 0.2274135f));
  assert(CheckEq(y(0, 1), 0.15243983f));
  assert(CheckEq(y(0, 2), 0.41437442f));
  assert(CheckEq(y(0, 3), 0.20577225f));
}


void TestLogSoftmaxLayer() {
  // Create the Softmax layer
  LogSoftmaxLayer softmax;

  float x_data[] = {
    0.6926, 0.5312, 0.3551,
    0.1014, 0.4569, 0.6337,
    0.5657, 0.8495, 0.8210,
    0.0483, 0.1684, 0.9234
  };
  SubMatrix<float> x(x_data, 4, 3, 3);
  Matrix<float> y;
  softmax.Propagate(x, &y);

  // Check results
  // Check results
  assert(y.NumCols() == 3 && y.NumRows() == 4);
  assert(CheckVector(y.Row(0), {-0.9418, -1.1032, -1.2793}));
  assert(CheckVector(y.Row(1), {-1.4182, -1.0627, -0.8859}));
  assert(CheckVector(y.Row(2), {-1.2862, -1.0024, -1.0309}));
  assert(CheckVector(y.Row(3), {-1.5100, -1.3899, -0.6349}));
}

void TestReLULayer() {
  // Create the ReLU layer
  ReLULayer relu;

  // Propagation
  float x_data[] = {0.3, -0.1, 0.9, 0.2};
  SubMatrix<float> x(x_data, 1, 4, 4);
  Matrix<float> y;
  relu.Propagate(x, &y);

  // Check results
  assert(y.NumCols() == 4);
  assert(y.NumRows() == 1);
  assert(CheckEq(y(0, 0), 0.3f));
  assert(CheckEq(y(0, 1), 0.0f));
  assert(CheckEq(y(0, 2), 0.9f));
  assert(CheckEq(y(0, 3), 0.2f));
}

void TestNormalizeLayer() {
  // Create the normalize layer
  NormalizeLayer normalize;

  // Propagation
  float x_data[] = {0.3, -0.1, 0.9, 0.2};
  SubMatrix<float> x(x_data, 1, 4, 4);
  Matrix<float> y;
  normalize.Propagate(x, &y);

  // Check results
  double sum = 0.0;
  for (int d = 0; d < 4; ++d) {
    sum += y(0, d) * y(0, d);
  }
  assert(fabs(sum - 4.0) < 0.0001);
}

void TestBatchNormLayer() {
  BatchNormLayer batch_norm(1e-5f);

  float x_data[] = {
    0.6926, 0.5312, 0.3551,
    0.1014, 0.4569, 0.6337,
    0.5657, 0.8495, 0.8210,
    0.0483, 0.1684, 0.9234
  };
  SubMatrix<float> x(x_data, 4, 3, 3);
  Matrix<float> y;
  batch_norm.Propagate(x, &y);

  // Check results
  assert(y.NumCols() == 3 && y.NumRows() == 4);
  assert(CheckVector(y.Row(0), {1.2105,  0.1228, -1.5185}));
  assert(CheckVector(y.Row(1), {-0.8905, -0.1840, -0.2297}));
  assert(CheckVector(y.Row(2), {0.7593,  1.4357,  0.6372}));
  assert(CheckVector(y.Row(3), {-1.0793, -1.3745,  1.1110}));
}

void TestNarrowLayer() {
  NarrowLayer narrow_layer(1, 2);

  // Matrix W
  float W_data[] = {
    0.1, 0.8, 0.9,
    0.4, 0.2, 0.7,
    0.2, 0.1, 0.1,
    0.4, 0.3, 0.2,
    0.5, 0.6, 0.7
  };
  SubMatrix<float> W(W_data, 5, 3, 3);
  Matrix<float> y;
  narrow_layer.Propagate(W, &y);

  // Check results
  assert(y.NumCols() == 3 && y.NumRows() == 2);
  assert(CheckVector(y.Row(0), {0.4, 0.2, 0.7}));
  assert(CheckVector(y.Row(1), {0.2, 0.1, 0.1}));

  // Check smaller matrix
  SubMatrix<float> W2(W_data, 3, 3, 3);
  narrow_layer.Propagate(W2, &y);

  // Check results
  assert(y.NumCols() == 3 && y.NumRows() == 3);
  assert(CheckVector(y.Row(0), {0.1, 0.8, 0.9}));
  assert(CheckVector(y.Row(1), {0.4, 0.2, 0.7}));
  assert(CheckVector(y.Row(2), {0.2, 0.1, 0.1}));
}

int main() {
  TestLinearLayer();
  TestSoftmaxLayer();
  TestLogSoftmaxLayer();
  TestReLULayer();
  TestNormalizeLayer();
  TestSpliceLayer();
  TestBatchNormLayer();
  TestNarrowLayer();
  return 0;
}
