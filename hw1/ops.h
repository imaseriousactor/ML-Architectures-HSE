#pragma once

#include <cstddef>
#include <cstdint>

namespace simd_math {

size_t simd_lanes_f32();
enum class BinOp { Add, Sub, Mul, Div };

void binary(const int32_t *a, const int32_t *b, int32_t *out, size_t n, BinOp op);
void binary(const float *a, const float *b, float *out, size_t n, BinOp op);
void binary(const double *a, const double *b, double *out, size_t n, BinOp op);

void binary_parallel(const int32_t *a, const int32_t *b, int32_t *out, size_t n, BinOp op);
void binary_parallel(const float *a, const float *b, float *out, size_t n, BinOp op);
void binary_parallel(const double *a, const double *b, double *out, size_t n, BinOp op);

template <typename T> struct SumType { using type = T; };
template <> struct SumType<int32_t> { using type = int64_t; };
template <typename T> using sum_t = typename SumType<T>::type;

int64_t sum(const int32_t *a, size_t n);
float sum(const float *a, size_t n);
double sum(const double *a, size_t n);

int64_t sum_parallel(const int32_t *a, size_t n);
float sum_parallel(const float *a, size_t n);
double sum_parallel(const double *a, size_t n);

double mean(const int32_t *a, size_t n);
double mean(const float *a, size_t n);
double mean(const double *a, size_t n);

double mean_parallel(const int32_t *a, size_t n);
double mean_parallel(const float *a, size_t n);
double mean_parallel(const double *a, size_t n);

enum class Act { Silu, Gelu };

void activation(const float *in, float *out, size_t n, Act act);
void activation(const double *in, double *out, size_t n, Act act);

void activation_parallel(const float *in, float *out, size_t n, Act act);
void activation_parallel(const double *in, double *out, size_t n, Act act);

void matmul(const int32_t *a, const int32_t *b, int32_t *c, size_t n);
void matmul(const float *a, const float *b, float *c, size_t n);
void matmul(const double *a, const double *b, double *c, size_t n);

void matmul_parallel(const int32_t *a, const int32_t *b, int32_t *c, size_t n);
void matmul_parallel(const float *a, const float *b, float *c, size_t n);
void matmul_parallel(const double *a, const double *b, double *c, size_t n);

} // namespace simd_math