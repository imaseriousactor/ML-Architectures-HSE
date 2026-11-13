#if defined(OPS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef OPS_INL_H_
#undef OPS_INL_H_
#else
#define OPS_INL_H_
#endif

// It is fine to #include normal or *-inl headers.
#include "hwy/highway.h"
#include "ops.h"

HWY_BEFORE_NAMESPACE();
namespace simd_math {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_MAYBE_UNUSED size_t lanes_f32() {
  const hn::ScalableTag<float> d;
  return hn::Lanes(d);
}

struct AddOp {
  template <class D, class V> V operator()(D, V a, V b) const { return hn::Add(a, b); }
  template <class T> T operator()(T a, T b) const { return a + b; }
};
struct SubOp {
  template <class D, class V> V operator()(D, V a, V b) const { return hn::Sub(a, b); }
  template <class T> T operator()(T a, T b) const { return a - b; }
};
struct MulOp {
  template <class D, class V> V operator()(D, V a, V b) const { return hn::Mul(a, b); }
  template <class T> T operator()(T a, T b) const { return a * b; }
};
struct DivOp {
  template <class D, class V> V operator()(D, V a, V b) const { return hn::Div(a, b); }
  template <class T> T operator()(T a, T b) const { return a / b; }
};

template <typename T, class Op>
void binary_op(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b, T *HWY_RESTRICT out, size_t n, Op op) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  size_t i = 0;
  for (; i + N <= n; i += N) {
    const auto va = hn::LoadU(d, a + i);
    const auto vb = hn::LoadU(d, b + i);
    hn::StoreU(op(d, va, vb), d, out + i);
  }
  for (; i < n; ++i) {
    out[i] = op(a[i], b[i]);
  }
}

template <typename T>
HWY_MAYBE_UNUSED void binary_kernel(const T *a, const T *b, T *out, size_t n, BinOp op) {
  switch (op) {
    case BinOp::Add: binary_op(a, b, out, n, AddOp{}); break;
    case BinOp::Sub: binary_op(a, b, out, n, SubOp{}); break;
    case BinOp::Mul: binary_op(a, b, out, n, MulOp{}); break;
    case BinOp::Div: binary_op(a, b, out, n, DivOp{}); break;
  }
}

template <typename T>
HWY_MAYBE_UNUSED sum_t<T> sum_kernel(const T *HWY_RESTRICT a, size_t n) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  size_t i = 0;
  sum_t<T> total = 0;

  if constexpr (std::is_same_v<T, int32_t>) {
    const hn::Repartition<int64_t, decltype(d)> d64;
    auto acc = hn::Zero(d64);
    for (; i + N <= n; i += N) {
      const auto v = hn::LoadU(d, a + i);
      acc = hn::Add(acc, hn::PromoteLowerTo(d64, v));
      acc = hn::Add(acc, hn::PromoteUpperTo(d64, v));
    }
    total = hn::ReduceSum(d64, acc);
  } else {
    auto acc = hn::Zero(d);
    for (; i + N <= n; i += N) {
      acc = hn::Add(acc, hn::LoadU(d, a + i));
    }
    total = hn::ReduceSum(d, acc);
  }

  for (; i < n; ++i) {
    total += a[i];
  }
  return total;
}

template <class D, class V>
HWY_INLINE V Tanh(D d, V x) {
  using T = hn::TFromD<D>;
  const auto lim = hn::Set(d, T(4.97));
  x = hn::Min(lim, hn::Max(hn::Neg(lim), x));
  const auto x2 = hn::Mul(x, x);

  auto num = hn::Add(x2, hn::Set(d, T(378)));
  num = hn::MulAdd(num, x2, hn::Set(d, T(17325)));
  num = hn::MulAdd(num, x2, hn::Set(d, T(135135)));
  num = hn::Mul(num, x);

  auto den = hn::MulAdd(hn::Set(d, T(28)), x2, hn::Set(d, T(3150)));
  den = hn::MulAdd(den, x2, hn::Set(d, T(62370)));
  den = hn::MulAdd(den, x2, hn::Set(d, T(135135)));

  return hn::Div(num, den);
}

  template <class D, class V>
HWY_INLINE V Sigmoid(D d, V x) {
  using T = hn::TFromD<D>;
  const auto half = hn::Set(d, T(0.5));
  return hn::MulAdd(half, Tanh(d, hn::Mul(half, x)), half);
}

struct SiluOp {
  template <class D, class V> V operator()(D d, V x) const {
    return hn::Mul(x, Sigmoid(d, x));
  }
};

struct GeluOp {
  template <class D, class V> V operator()(D d, V x) const {
    using T = hn::TFromD<D>;
    const auto k = hn::Set(d, T(0.7978845608028654));
    const auto c = hn::Set(d, T(0.044715));
    const auto half = hn::Set(d, T(0.5));
    const auto x3 = hn::Mul(hn::Mul(x, x), x);
    const auto t = Tanh(d, hn::Mul(k, hn::MulAdd(c, x3, x)));
    return hn::Mul(hn::Mul(half, x), hn::Add(hn::Set(d, T(1)), t));
  }
};

template <typename T, class Op>
void unary_op(const T *HWY_RESTRICT in, T *HWY_RESTRICT out, size_t n, Op op) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  size_t i = 0;
  for (; i + N <= n; i += N) {
    hn::StoreU(op(d, hn::LoadU(d, in + i)), d, out + i);
  }
  if (i < n) {
    const size_t rest = n - i;
    const auto v = hn::LoadN(d, in + i, rest);
    hn::StoreN(op(d, v), d, out + i, rest);
  }
}

template <typename T>
HWY_MAYBE_UNUSED void activation_kernel(const T *in, T *out, size_t n, Act act) {
  switch (act) {
    case Act::Silu: unary_op(in, out, n, SiluOp{}); break;
    case Act::Gelu: unary_op(in, out, n, GeluOp{}); break;
  }
}

template <typename T>
HWY_MAYBE_UNUSED void matmul_kernel(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                                    T *HWY_RESTRICT c, size_t rows, size_t n) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);

  for (size_t i = 0; i < rows * n; ++i) {
    c[i] = 0;
  }

  for (size_t i = 0; i < rows; ++i) {
    T *HWY_RESTRICT crow = c + i * n;
    for (size_t k = 0; k < n; ++k) {
      const T a_ik = a[i * n + k];
      const T *HWY_RESTRICT brow = b + k * n;
      const auto va = hn::Set(d, a_ik);

      size_t j = 0;
      for (; j + N <= n; j += N) {
        auto acc = hn::LoadU(d, crow + j);
        const auto vb = hn::LoadU(d, brow + j);
        acc = hn::MulAdd(va, vb, acc);
        hn::StoreU(acc, d, crow + j);
      }
      for (; j < n; ++j) {
        crow[j] += a_ik * brow[j];
      }
    }
  }
}

} // namespace HWY_NAMESPACE
} // namespace simd_math
HWY_AFTER_NAMESPACE();
#endif