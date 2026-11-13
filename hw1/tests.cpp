#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "data_generators.hpp"
#include "ops.h"
#include "scalar.h"

using simd_math::BinOp;

const std::vector<size_t> sizes = {1, 7, 8, 17, 1000, 1023, 4096, 100003};

TEST(Setup, HighwayWorks) {
  EXPECT_GE(simd_math::simd_lanes_f32(), 1u);
  std::cout << "float lanes: " << simd_math::simd_lanes_f32() << "\n";
}

template <typename T>
void check_arrays(const std::vector<T> &expected, const std::vector<T> &actual, double tol) {
  for (size_t i = 0; i < expected.size(); ++i) {
    double e = expected[i];
    double a = actual[i];
    ASSERT_NEAR(e, a, tol * std::max(1.0, std::abs(e))) << "i = " << i;
  }
}

template <typename T> class BinaryTest : public ::testing::Test {};
using Types = ::testing::Types<int32_t, float, double>;
TYPED_TEST_SUITE(BinaryTest, Types);

template <typename T, class Op>
void check_binary(BinOp op, Op scalar_op) {
  double tol = std::is_integral_v<T> ? 0 : 1e-6;
  for (size_t n : sizes) {
    auto [a, b] = make_test_data<T>(42, n);
    for (size_t i = 0; i < n; ++i) b[i] += 1;

    std::vector<T> expected(n), actual(n), parallel(n);
    scalar::binary(a.data(), b.data(), expected.data(), n, scalar_op);
    simd_math::binary(a.data(), b.data(), actual.data(), n, op);
    simd_math::binary_parallel(a.data(), b.data(), parallel.data(), n, op);

    check_arrays(expected, actual, tol);
    check_arrays(expected, parallel, tol);
  }
}

TYPED_TEST(BinaryTest, Add) { check_binary<TypeParam>(BinOp::Add, std::plus<TypeParam>()); }
TYPED_TEST(BinaryTest, Sub) { check_binary<TypeParam>(BinOp::Sub, std::minus<TypeParam>()); }
TYPED_TEST(BinaryTest, Mul) { check_binary<TypeParam>(BinOp::Mul, std::multiplies<TypeParam>()); }
TYPED_TEST(BinaryTest, Div) { check_binary<TypeParam>(BinOp::Div, std::divides<TypeParam>()); }

template <typename T> class ReduceTest : public ::testing::Test {};
TYPED_TEST_SUITE(ReduceTest, Types);

TYPED_TEST(ReduceTest, SumAndMean) {
  using T = TypeParam;
  double tol = 1e-10;
  if (std::is_same_v<T, float>) tol = 1e-4;
  if (std::is_integral_v<T>) tol = 0;

  for (size_t n : sizes) {
    auto [a, b] = make_test_data<T>(42, n);

    double expected = 0;
    for (size_t i = 0; i < n; ++i) expected += a[i];
    double expected_mean = expected / n;

    EXPECT_NEAR(expected, simd_math::sum(a.data(), n), tol * expected) << "n = " << n;
    EXPECT_NEAR(expected, simd_math::sum_parallel(a.data(), n), tol * expected) << "n = " << n;
    EXPECT_NEAR(expected_mean, simd_math::mean(a.data(), n), tol * expected_mean + 1e-12);
    EXPECT_NEAR(expected_mean, simd_math::mean_parallel(a.data(), n), tol * expected_mean + 1e-12);
  }
}

template <typename T> class ActivationTest : public ::testing::Test {};
using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ActivationTest, FloatTypes);

template <typename T, class Op>
void check_activation(simd_math::Act act, Op scalar_op) {
  for (size_t n : sizes) {
    auto [x, unused] = make_test_data<T>(42, n);
    for (size_t i = 0; i < n; ++i) x[i] = 20 * x[i] - 10;

    std::vector<T> expected(n), actual(n), parallel(n);
    scalar::unary(x.data(), expected.data(), n, scalar_op);
    simd_math::activation(x.data(), actual.data(), n, act);
    simd_math::activation_parallel(x.data(), parallel.data(), n, act);

    for (size_t i = 0; i < n; ++i) {
      double tol = 1e-4 * std::max(1.0, std::abs(double(x[i])));
      ASSERT_NEAR(expected[i], actual[i], tol) << "x = " << x[i];
      ASSERT_NEAR(expected[i], parallel[i], tol) << "x = " << x[i];
    }
  }
}

TYPED_TEST(ActivationTest, Silu) { check_activation<TypeParam>(simd_math::Act::Silu, scalar::Silu{}); }
TYPED_TEST(ActivationTest, Gelu) { check_activation<TypeParam>(simd_math::Act::Gelu, scalar::Gelu{}); }

template <typename T> class MatmulTest : public ::testing::Test {};
TYPED_TEST_SUITE(MatmulTest, Types);

TYPED_TEST(MatmulTest, MatchesScalar) {
  using T = TypeParam;
  for (size_t n : {1, 3, 8, 17, 64, 100, 129}) {
    std::vector<T> a(n * n), b(n * n);
    for (size_t i = 0; i < n * n; ++i) {
      a[i] = T(i % 7) - 3;
      b[i] = T(i % 5) - 2;
    }

    std::vector<T> expected(n * n), actual(n * n), parallel(n * n);
    scalar::matmul(a.data(), b.data(), expected.data(), n);
    simd_math::matmul(a.data(), b.data(), actual.data(), n);
    simd_math::matmul_parallel(a.data(), b.data(), parallel.data(), n);

    check_arrays(expected, actual, 0);
    check_arrays(expected, parallel, 0);
  }
}