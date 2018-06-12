// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SOLVE_KERNEL_HPP)
#define SOLVE_KERNEL_HPP

#include <cstdint>
#include <vector>
#include <fp/fp_blas.hpp>

//using real_t = float;
using real_t = double;

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

void triangular_solve(const bool transpose, const std::size_t n, const fp_t* a, real_t* x, const std::size_t bs = 32, std::vector<real_t>* buffer = nullptr);

#endif
