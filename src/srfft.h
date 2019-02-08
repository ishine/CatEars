// 2016-12-16

#ifndef POCKETKALDI_SRFFT_H_
#define POCKETKALDI_SRFFT_H_

#include <stdbool.h>
#include "ce_stt.h"

namespace pocketkaldi {

// 
class SRFFT {
 public:
  // N is the number of complex points (must be a power of two, or this
  // will crash).  Note that the constructor does some work so it's best to
  // initialize the object once and do the computation many times.
  SRFFT(int N);
  ~SRFFT();

  // This version of Compute is const; it operates on an array of size N*2
  // containing [ r0 im0 r1 im1 ... ], but it uses the argument "temp_buffer" as
  // temporary storage instead of a class-member variable.  It will allocate it if
  // needed.
  // `buffer` needs at least N float spaces
  void Compute(
      float *data,
      int data_size,
      bool forward,
      float *buffer,
      int buffer_size) const;

 private:
  int N_;
  int logn_;

  // brseed is Evans' seed table, ref:  (Ref: D. M. W.
  // Evans, "An improved digit-reversal permutation algorithm ...",
  // IEEE Trans. ASSP, Aug. 1987, pp. 1120-1125).
  int *brseed_;

  // Tables of butterfly coefficients.
  float **tab_;

  void BitReversePermute(float *x, int logn) const;
  void ComplexFFTCompute2(float *xr, float *xi, bool forward) const;
  void ComplexfftComputeRecursive(float *xr, float *xi, int logn) const;
  void ComputeTable();
  void ComplexFFTCompute(
      float *x,
      int xsize,
      bool forward,
      float *buffer,
      int buffer_size) const;
};

}  // namespace pocketkaldi



#endif  // POCKETKALDI_SRFFT_H_
