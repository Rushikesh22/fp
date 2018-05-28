// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(MATRIX_VECTOR_KERNEL_HPP)
#define MATRIX_VECTOR_KERNEL_HPP

#include <cstdint>
#include <vector>
#include <fp/fp_blas.hpp>

using real_t = float;
//using real_t = double;

#if defined(_BE)
static constexpr std::int32_t BE = _BE;
#else
static constexpr std::int32_t BE = fw::fp<real_t>::default_bits_exponent();
#endif

#if defined(_BM)
static constexpr std::int32_t BM = _BM;
#else
static constexpr std::int32_t BM = fw::fp<real_t>::default_bits_mantissa();
#endif

using fp_t = typename fw::fp<real_t>::template format<BE, BM>::type;

void full_matrix_vector(const bool transpose, const std::size_t m, const std::size_t n, const real_t alpha, const fp_t* a, const real_t* x, const real_t beta, real_t* y, const std::size_t bs = 32, std::vector<real_t>* buffer = nullptr);

#endif
