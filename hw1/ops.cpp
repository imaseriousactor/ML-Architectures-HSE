#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops.cpp"
#include <hwy/foreach_target.h>

#include "ops-inl.h"
#include <hwy/highway.h>

#if HWY_ONCE
#include "ops.h"
#include <oneapi/tbb.h>

namespace simd_math {

HWY_EXPORT(lanes_f32);

size_t simd_lanes_f32() { 
  return HWY_DYNAMIC_DISPATCH(lanes_f32)(); 
}

template <typename T>
void binary_impl(const T *a, const T *b, T *out, size_t n, BinOp op) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(binary_kernel<T>)(a, b, out, n, op);
}

void binary(const int32_t *a, const int32_t *b, int32_t *out, size_t n, BinOp op) { binary_impl(a, b, out, n, op); }
void binary(const float *a, const float *b, float *out, size_t n, BinOp op) { binary_impl(a, b, out, n, op); }
void binary(const double *a, const double *b, double *out, size_t n, BinOp op) { binary_impl(a, b, out, n, op); }

template <typename T>
constexpr size_t grain_for() { return 32768 / sizeof(T); }

template <class F>
void parallel_blocks(size_t n, size_t grain, F &&kernel) {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, n, grain),
      [&](const oneapi::tbb::blocked_range<size_t> &r) {
        kernel(r.begin(), r.end() - r.begin());
      });
}

template <typename T>
void binary_parallel_impl(const T *a, const T *b, T *out, size_t n, BinOp op) {
  parallel_blocks(n, grain_for<T>(), [=](size_t begin, size_t len) {
    binary_impl(a + begin, b + begin, out + begin, len, op);
  });
}

void binary_parallel(const int32_t *a, const int32_t *b, int32_t *out, size_t n, BinOp op) { binary_parallel_impl(a, b, out, n, op); }
void binary_parallel(const float *a, const float *b, float *out, size_t n, BinOp op) { binary_parallel_impl(a, b, out, n, op); }
void binary_parallel(const double *a, const double *b, double *out, size_t n, BinOp op) { binary_parallel_impl(a, b, out, n, op); }

template <typename T>
sum_t<T> sum_impl(const T *a, size_t n) {
  HWY_EXPORT_T(SumTable, sum_kernel<T>);
  return HWY_DYNAMIC_DISPATCH_T(SumTable)(a, n);
}

template <typename T>
sum_t<T> sum_parallel_impl(const T *a, size_t n) {
  return oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<size_t>(0, n, grain_for<T>()),
      sum_t<T>(0),
      [a](const oneapi::tbb::blocked_range<size_t> &r, sum_t<T> acc) {
        return acc + sum_impl(a + r.begin(), r.end() - r.begin());
      },
      std::plus<sum_t<T>>());
}

int64_t sum(const int32_t *a, size_t n) { return sum_impl(a, n); }
float sum(const float *a, size_t n) { return sum_impl(a, n); }
double sum(const double *a, size_t n) { return sum_impl(a, n); }

int64_t sum_parallel(const int32_t *a, size_t n) { return sum_parallel_impl(a, n); }
float sum_parallel(const float *a, size_t n) { return sum_parallel_impl(a, n); }
double sum_parallel(const double *a, size_t n) { return sum_parallel_impl(a, n); }

double mean(const int32_t *a, size_t n) { return double(sum_impl(a, n)) / n; }
double mean(const float *a, size_t n) { return double(sum_impl(a, n)) / n; }
double mean(const double *a, size_t n) { return sum_impl(a, n) / n; }

double mean_parallel(const int32_t *a, size_t n) { return double(sum_parallel_impl(a, n)) / n; }
double mean_parallel(const float *a, size_t n) { return double(sum_parallel_impl(a, n)) / n; }
double mean_parallel(const double *a, size_t n) { return sum_parallel_impl(a, n) / n; }

template <typename T>
void activation_impl(const T *in, T *out, size_t n, Act act) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(activation_kernel<T>)(in, out, n, act);
}

template <typename T>
void activation_parallel_impl(const T *in, T *out, size_t n, Act act) {
  parallel_blocks(n, grain_for<T>(), [=](size_t begin, size_t len) {
    activation_impl(in + begin, out + begin, len, act);
  });
}

void activation(const float *in, float *out, size_t n, Act act) { activation_impl(in, out, n, act); }
void activation(const double *in, double *out, size_t n, Act act) { activation_impl(in, out, n, act); }

void activation_parallel(const float *in, float *out, size_t n, Act act) { activation_parallel_impl(in, out, n, act); }
void activation_parallel(const double *in, double *out, size_t n, Act act) { activation_parallel_impl(in, out, n, act); }

template <typename T>
void matmul_rows(const T *a, const T *b, T *c, size_t rows, size_t n) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(matmul_kernel<T>)(a, b, c, rows, n);
}

template <typename T>
void matmul_impl(const T *a, const T *b, T *c, size_t n) {
  matmul_rows(a, b, c, n, n);
}

template <typename T>
void matmul_parallel_impl(const T *a, const T *b, T *c, size_t n) {
  parallel_blocks(n, 8, [=](size_t row_begin, size_t rows) {
    matmul_rows(a + row_begin * n, b, c + row_begin * n, rows, n);
  });
}

void matmul(const int32_t *a, const int32_t *b, int32_t *c, size_t n) { matmul_impl(a, b, c, n); }
void matmul(const float *a, const float *b, float *c, size_t n) { matmul_impl(a, b, c, n); }
void matmul(const double *a, const double *b, double *c, size_t n) { matmul_impl(a, b, c, n); }

void matmul_parallel(const int32_t *a, const int32_t *b, int32_t *c, size_t n) { matmul_parallel_impl(a, b, c, n); }
void matmul_parallel(const float *a, const float *b, float *c, size_t n) { matmul_parallel_impl(a, b, c, n); }
void matmul_parallel(const double *a, const double *b, double *c, size_t n) { matmul_parallel_impl(a, b, c, n); }

} // namespace simd_math
#endif // HWY_ONCE