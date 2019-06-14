// gemm.cc -- Created at 2017-05-29

#include <math.h>
#include <random>
#include <tuple>
#include "matrix.h"

using pocketkaldi::Matrix;
using pocketkaldi::MatrixBase;
using pocketkaldi::SimpleMatMat;
using pocketkaldi::MatMat;
using pocketkaldi::MatMat_U8U8F32;
using pocketkaldi::QuantizationParams;
using pocketkaldi::Quantize;

void FindMinMax(const MatrixBase<float> &src, float *pmin, float *pmax) {
  float min = std::numeric_limits<float>::max(),
        max = std::numeric_limits<float>::min();

  for (int i = 0; i < src.NumCols() * src.NumRows(); ++i) {
    float val = src.Data()[i];
    if (val > max) {
      max = val;
    }
    if (val < min) {
      min = val;
    }
  }

  *pmin = min;
  *pmax = max;
}

// Returns the maximun difference of dimensions
float CompareMatrix(
    const MatrixBase<float> &A,
    const MatrixBase<float> &B) {
  assert(A.NumCols() == B.NumCols());
  assert(A.NumRows() == B.NumRows());

  float max_diff = 0.0f;
  for (int row_idx = 0; row_idx < A.NumRows(); ++row_idx) {
    for (int col_idx = 0; col_idx < A.NumCols(); ++col_idx) {
      float diff = fabs(A(row_idx, col_idx) - B(row_idx, col_idx));
      if (diff > max_diff) max_diff = diff;
    }
  }

  return max_diff;
}

void PrintMatrix(const MatrixBase<float> &m) {
  for (int row = 0; row < m.NumRows(); ++row) {
    for (int col = 0; col < m.NumCols(); ++col) {
      printf("%f ", m(row, col));
    }
  }
}

// Generate a random matrix that filled with random numbers between (min, max]
float GenerateRandomMatrix(
    int num_row, int num_col, 
    float min, float max, Matrix<float> *mat) {
  mat->Resize(num_row, num_col, Matrix<float>::kUndefined);

  // Fill random numbers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);

  for (int row = 0; row < mat->NumRows(); ++row) {
    for (int col = 0; col < mat->NumCols(); ++col) {
      (*mat)(row, col) = dis(gen);
    }
  }
}

void TestSgemm() {
  Matrix<float> A;
  Matrix<float> B;
  Matrix<float> C;
  Matrix<float> CRef;

  std::vector<std::tuple<int, int, int>> test_sizes {
    std::make_tuple(5, 3, 2),
    std::make_tuple(100, 100, 1),
    std::make_tuple(1024, 1024, 80),
    std::make_tuple(121, 233, 17)
  };
  for (const std::tuple<int, int, int> &test_size : test_sizes) {
    int m = std::get<0>(test_size);
    int n = std::get<1>(test_size);
    int k = std::get<2>(test_size);

    GenerateRandomMatrix(m, k, -0.5, 0.5, &A);
    GenerateRandomMatrix(k, n, 1, 2, &B);
    C.Resize(m, n);
    CRef.Resize(m, n);

    SimpleMatMat(A, B, &CRef);
    MatMat(A, B, &C);

    assert(CompareMatrix(C, CRef) < 0.01);

    // Check 8-bit gemm  
    Matrix<u_int8_t> A_8bit, B_8bit;
    QuantizationParams quant_params_A, quant_params_B;
    Quantize(A, &A_8bit, &quant_params_A);
    Quantize(B, &B_8bit, &quant_params_B);

    C.SetZero();
    MatMat_U8U8F32(A_8bit, quant_params_A, B_8bit, quant_params_B, &C);

    float min, max;
    FindMinMax(C, &min, &max);
    float max_diff = CompareMatrix(C, CRef);
    printf("min = %f, max = %f, max_diff = %f\n", min, max, max_diff);

    assert(max_diff / (max - min) < 0.01);
  }
}

int main() {
  TestSgemm();
  return 0;
}
