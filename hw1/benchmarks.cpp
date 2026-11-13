#include "bench_utils.hpp"
#include "data_generators.hpp"

#include "ops.h"
#include "scalar.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace bench;
using simd_math::Act;
using simd_math::BinOp;

enum class Impl { Scalar, Simd, Parallel };
const char *impl_name(Impl impl) {
  switch (impl) {
    case Impl::Scalar: return "scalar";
    case Impl::Simd: return "simd";
    case Impl::Parallel: return "simd_tbb";
  }
  return "";
}

template <typename T> const char *type_name();
template <> const char *type_name<int32_t>() { return "int32"; }
template <> const char *type_name<float>() { return "float32"; }
template <> const char *type_name<double>() { return "float64"; }

template <typename T>
void run_binary(benchmark::State &state, BinOp op, Impl impl) {
  const size_t n = state.range(0);
  auto [a, b] = make_test_data<T>(seed, n);
  for (auto &v : b) v += 1;
  std::vector<T> out(n);

  for (auto _ : state) {
    if (impl == Impl::Scalar) {
      switch (op) {
        case BinOp::Add: scalar::binary(a.data(), b.data(), out.data(), n, std::plus<T>{}); break;
        case BinOp::Sub: scalar::binary(a.data(), b.data(), out.data(), n, std::minus<T>{}); break;
        case BinOp::Mul: scalar::binary(a.data(), b.data(), out.data(), n, std::multiplies<T>{}); break;
        case BinOp::Div: scalar::binary(a.data(), b.data(), out.data(), n, std::divides<T>{}); break;
      }
    } else if (impl == Impl::Simd) {
      simd_math::binary(a.data(), b.data(), out.data(), n, op);
    } else {
      simd_math::binary_parallel(a.data(), b.data(), out.data(), n, op);
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * 3 * n * sizeof(T));
}

template <typename T>
void run_reduce(benchmark::State &state, bool is_mean, Impl impl) {
  const size_t n = state.range(0);
  auto [a, unused] = make_test_data<T>(seed, n);

  for (auto _ : state) {
    double result = 0;
    if (impl == Impl::Scalar) {
      result = double(scalar::sum<simd_math::sum_t<T>>(a.data(), n));
      if (is_mean) result /= n;
    } else if (impl == Impl::Simd) {
      result = is_mean ? simd_math::mean(a.data(), n) : double(simd_math::sum(a.data(), n));
    } else {
      result = is_mean ? simd_math::mean_parallel(a.data(), n) : double(simd_math::sum_parallel(a.data(), n));
    }
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * n * sizeof(T));
}

template <typename T>
void run_activation(benchmark::State &state, Act act, Impl impl) {
  const size_t n = state.range(0);
  auto [x, unused] = make_test_data<T>(seed, n);
  for (auto &v : x) v = 20 * v - 10;
  std::vector<T> out(n);

  for (auto _ : state) {
    if (impl == Impl::Scalar) {
      if (act == Act::Silu) scalar::unary(x.data(), out.data(), n, scalar::Silu{});
      else scalar::unary(x.data(), out.data(), n, scalar::Gelu{});
    } else if (impl == Impl::Simd) {
      simd_math::activation(x.data(), out.data(), n, act);
    } else {
      simd_math::activation_parallel(x.data(), out.data(), n, act);
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * 2 * n * sizeof(T));
}

template <typename T>
void run_matmul(benchmark::State &state, Impl impl) {
  const size_t n = state.range(0);
  auto [a, b] = make_test_data<T>(seed, n * n);
  if constexpr (std::is_integral_v<T>) {
    for (auto &v : a) v %= 100;
    for (auto &v : b) v %= 100;
  }
  std::vector<T> c(n * n);

  for (auto _ : state) {
    if (impl == Impl::Scalar) scalar::matmul(a.data(), b.data(), c.data(), n);
    else if (impl == Impl::Simd) simd_math::matmul(a.data(), b.data(), c.data(), n);
    else simd_math::matmul_parallel(a.data(), b.data(), c.data(), n);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  state.counters["FLOPS"] = benchmark::Counter(2.0 * n * n * n,
                                               benchmark::Counter::kIsIterationInvariantRate,
                                               benchmark::Counter::OneK::kIs1000);
}

template <typename T>
void register_for_type() {
  const std::string t = type_name<T>();
  const Impl impls[] = {Impl::Scalar, Impl::Simd, Impl::Parallel};
  const std::pair<const char *, BinOp> binops[] = {
      {"Add", BinOp::Add}, {"Sub", BinOp::Sub}, {"Mul", BinOp::Mul}, {"Div", BinOp::Div}};

  for (Impl impl : impls) {
    const std::string suffix = "/" + t + "/" + impl_name(impl);

    for (auto [name, op] : binops) {
      benchmark::RegisterBenchmark(name + suffix, run_binary<T>, op, impl)
          ->Apply(problems)->Unit(benchmark::kMicrosecond);
    }
    benchmark::RegisterBenchmark("Sum" + suffix, run_reduce<T>, false, impl)
        ->Apply(problems)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("Mean" + suffix, run_reduce<T>, true, impl)
        ->Apply(problems)->Unit(benchmark::kMicrosecond);

    if constexpr (std::is_floating_point_v<T>) {
      benchmark::RegisterBenchmark("SiLU" + suffix, run_activation<T>, Act::Silu, impl)
          ->Apply(problems)->Unit(benchmark::kMicrosecond);
      benchmark::RegisterBenchmark("GELU" + suffix, run_activation<T>, Act::Gelu, impl)
          ->Apply(problems)->Unit(benchmark::kMicrosecond);
    }

    benchmark::RegisterBenchmark("Matmul" + suffix, run_matmul<T>, impl)
        ->Apply(matmul_problems)->Unit(benchmark::kMicrosecond);
  }
}

static const int registered = [] {
  register_for_type<int32_t>();
  register_for_type<float>();
  register_for_type<double>();
  return 0;
}();