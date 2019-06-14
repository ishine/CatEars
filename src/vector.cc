
#include "vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

namespace pocketkaldi {

template<typename Real>
Vector<Real>::Vector(Vector<Real> &&v) {
  this->dim_ = v.dim_;
  this->data_ = v.data_;

  v.dim_ = 0;
  v.data_ = nullptr;
}

template<typename Real>
Vector<Real> &Vector<Real>::operator=(Vector<Real> &&v) {
  this->dim_ = v.dim_;
  this->data_ = v.data_;

  v.dim_ = 0;
  v.data_ = nullptr;

  return *this;
}

template<typename Real>
inline void Vector<Real>::Init(int dim) {
  assert(dim >= 0);
  if (dim == 0) {
    this->dim_ = 0;
    this->data_ = NULL;
    return;
  }
  int size;
  void *data;

  size = dim * sizeof(Real);

  if (posix_memalign(&data, 32, size) == 0) {
    this->data_ = static_cast<Real*>(data);
    this->dim_ = dim;
  } else {
    throw std::bad_alloc();
  }
}

/// Deallocates memory and sets object to empty vector.
template<typename Real>
void Vector<Real>::Destroy() {
  /// we need to free the data block if it was defined
  if (this->data_ != NULL) free(this->data_);
  this->data_ = NULL;
  this->dim_ = 0;
}

template<typename Real>
void VectorBase<Real>::Set(Real f) {
  // Why not use memset here?
  for (int i = 0; i < dim_; i++) { data_[i] = f; }
}

template<typename Real>
void VectorBase<Real>::SetZero() {
  memset(data_, 0, dim_ * sizeof(Real));
}

/// Copy data from another vector
template<typename Real>
void VectorBase<Real>::CopyFromVec(const VectorBase<Real> &v) {
  assert(Dim() == v.Dim());
  if (data_ != v.data_) {
    memcpy(this->data_, v.data_, dim_ * sizeof(Real));
  }
}

template<typename Real>
Real VectorBase<Real>::VecVec(const VectorBase<Real> &r) const {
  int r_dim = r.Dim();
  assert(r_dim == Dim());
  const Real *l_data = Data();
  const Real *r_data = r.Data();
  Real sum = 0.0;
  for (int i = 0; i < r_dim; i++) {
    sum += l_data[i] * r_data[i];
  }
  return sum;
}

template<typename Real>
void VectorBase<Real>::ApplySoftMax() {
  Real sum = 0;

  for (int i = 0; i < Dim(); ++i) {
    Real exp_d = expf((*this)(i));
    (*this)(i) = exp_d;
    sum += exp_d;
  }

  for (int i = 0; i < Dim(); ++i) {
    (*this)(i) /= sum;
  }
}

template<typename Real>
void VectorBase<Real>::ApplyLogSoftMax() {
  Real sum = 0;

  for (int i = 0; i < Dim(); ++i) {
    Real exp_d = exp((*this)(i));
    sum += exp_d;
  }
  Real logsum = log(sum);
  
  for (int i = 0; i < Dim(); ++i) {
    (*this)(i) -= logsum;
  }
}

template<typename Real>
void Vector<Real>::Swap(Vector<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->dim_, other->dim_);
}

template<typename Real>
void Vector<Real>::Resize(const int dim, int resize_type) {
  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    // nothing to copy.
    if (this->data_ == NULL || dim == 0) resize_type = kSetZero;  
    else if (this->dim_ == dim) { return; } // nothing to do.
    else {
      // set tmp to a vector of the desired size.
      Vector<Real> tmp(dim, kUndefined);
      if (dim > this->dim_) {
        memcpy(tmp.data_, this->data_, sizeof(Real)*this->dim_);
        memset(tmp.data_ + this->dim_, 0, sizeof(Real) * (dim-this->dim_));
      } else {
        memcpy(tmp.data_, this->data_, sizeof(Real)*dim);
      }
      tmp.Swap(this);
      // and now let tmp go out of scope, deleting what was in *this.
      return;
    }
  }
  // At this point, resize_type == kSetZero or kUndefined.

  if (this->data_ != NULL) {
    if (this->dim_ == dim) {
      if (resize_type == kSetZero) this->SetZero();
      return;
    } else {
      Destroy();
    }
  }
  Init(dim);
  if (resize_type == kSetZero) this->SetZero();
}

template<typename Real>
int VectorBase<Real>::ApplyFloor(Real floor_val) {
  int num_floored = 0;
  for (int i = 0; i < dim_; i++) {
    if (data_[i] < floor_val) {
      data_[i] = floor_val;
      num_floored++;
    }
  }
  return num_floored;
}

template<typename Real>
void VectorBase<Real>::ApplyLog() {
  for (int i = 0; i < dim_; i++) {
    assert(data_[i] >= 0.0);
    data_[i] = log(data_[i]);
  }
}

template<typename Real>
void VectorBase<Real>::ApplyPow(Real power) {
  for (int i = 0; i < dim_; i++) {
    data_[i] = pow(data_[i], power);
  }
}

template<typename Real>
void VectorBase<Real>::Scale(Real alpha) {
  for (int i = 0; i < dim_; i++) {
    data_[i] *= alpha;
  }
}


template<typename Real>
void VectorBase<Real>::Add(Real val) {
  for (int i = 0; i < dim_; i++) {
    data_[i] += val;
  }
}

template<typename Real>
void VectorBase<Real>::PrintDebug() {
  printf("vector: dim = %d, data = [", dim_);
  for (int i = 0; i < dim_; i++) {
    printf("%s, ", std::to_string(data_[i]).c_str());
  }
  puts("]");
}

template<typename Real>
void VectorBase<Real>::MulElements(const VectorBase<Real> &v) {
  assert(v.Dim() == Dim() && "MulElements: vector size mismatch");
  for (int i = 0; i < dim_; i++) {
    data_[i] *= v(i);
  }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyFromVec(const VectorBase<OtherReal> &other) {
  assert(dim_ == other.Dim());
  Real * __restrict__  ptr = data_;
  const OtherReal * __restrict__ other_ptr = other.Data();
  for (int i = 0; i < dim_; i++) {
    ptr[i] = other_ptr[i];
  }
}

template void VectorBase<float>::CopyFromVec(const VectorBase<double> &other);
template void VectorBase<double>::CopyFromVec(const VectorBase<float> &other);

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::AddVec(
    const Real alpha,
    const VectorBase<OtherReal> &v) {
  assert(dim_ == v.dim_);
  // remove __restrict__ if it causes compilation problems.
  Real *__restrict__ data = data_;
  OtherReal *__restrict__ other_data = v.data_;
  int dim = dim_;
  if (alpha != 1.0) {
    for (int i = 0; i < dim; i++) {
      data[i] += alpha * other_data[i];
    }
  } else {
    for (int i = 0; i < dim; i++) {
      data[i] += other_data[i];
    }
  }
}

template
void VectorBase<float>::AddVec(const float alpha, const VectorBase<double> &v);
template
void VectorBase<float>::AddVec(const float alpha, const VectorBase<float> &v);
template
void VectorBase<double>::AddVec(const double alpha, const VectorBase<float> &v);

template<typename Real>
Status Vector<Real>::Read(util::ReadableFile *fd) {
  static const char *kSectionName = "VEC0";
  Status status;

  // Read section name
  status = fd->ReadAndVerifyString(kSectionName);
  if (!status.ok()) return status;

  // Section size
  int32_t section_size;
  status = fd->ReadValue<int32_t>(&section_size);
  if (!status.ok()) return status;

  // Dimension
  int32_t dim;
  status = fd->ReadValue<int32_t>(&dim);
  if (!status.ok()) return status;
  if (dim * sizeof(Real) + 4 != section_size) {
    return Status::Corruption(util::Format(
        "section_size = {} * {} + 4 expected, but {} found: {}",
        dim,
        sizeof(Real),
        section_size,
        fd->filename()));
  }

  // Read data
  Resize(dim, kUndefined);
  status = fd->Read(Vector<Real>::Data(), dim * sizeof(Real));
  if (!status.ok()) return status;

  return Status::OK();
}

template class Vector<float>;
template class VectorBase<float>;
template class Vector<double>;
template class VectorBase<double>;
template class Vector<int32_t>;
template class VectorBase<int32_t>;
template class Vector<uint8_t>;
template class VectorBase<uint8_t>;

}  // namespace pocketkaldi
