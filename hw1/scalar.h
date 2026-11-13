#pragma once

#include <cstddef>
#include <functional>

namespace scalar {

template <typename T, class Op>
__attribute__((optimize("no-tree-vectorize")))
void binary(const T *a, const T *b, T *out, size_t n, Op op) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = op(a[i], b[i]);
  }
}

template <typename Acc, typename T>
__attribute__((optimize("no-tree-vectorize")))
Acc sum(const T *a, size_t n) {
  Acc s = 0;
  for (size_t i = 0; i < n; ++i) {
    s += a[i];
  }
  return s;
}

struct Silu {
  template <typename T> T operator()(T x) const { return x / (T(1) + std::exp(-x)); }
};
struct Gelu {
  template <typename T> T operator()(T x) const {
    const T k = T(0.7978845608028654);
    return T(0.5) * x * (T(1) + std::tanh(k * (x + T(0.044715) * x * x * x)));
  }
};

template <typename T, class Op>
__attribute__((optimize("no-tree-vectorize")))
void unary(const T *in, T *out, size_t n, Op op) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = op(in[i]);
  }
}

template <typename T>
__attribute__((optimize("no-tree-vectorize")))
void matmul(const T *a, const T *b, T *c, size_t n) {
  for (size_t i = 0; i < n * n; ++i) {
    c[i] = 0;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      const T a_ik = a[i * n + k];
      for (size_t j = 0; j < n; ++j) {
        c[i * n + j] += a_ik * b[k * n + j];
      }
    }
  }
}

} // namespace scalar