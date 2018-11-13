// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TRIANGULAR_SOLVE_KERNEL_HPP)
#define TRIANGULAR_SOLVE_KERNEL_HPP

#include <cstdint>
#include <vector>
#include <fp/fp_blas.hpp>

// fundamental real data type: 'float' or 'double'
//using real_t = float;
using real_t = double;

// number of bits to be used for exponent and mantissa
#if defined(_BE)
static constexpr std::uint32_t BE = _BE;
#else
static constexpr std::uint32_t BE = 0;
#endif

#if defined(_BM)
static constexpr std::uint32_t BM = _BM;
#else
static constexpr std::uint32_t BM = 15;
#endif

#if defined(_ROWMAJOR)
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::rowmajor;
#elif defined(_COLMAJOR)
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::colmajor;
#else
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::rowmajor;
#endif

static constexpr CBLAS_LAYOUT layout = (L == fw::blas::matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

// compressed matrix data type
#if defined(UPPER_MATRIX)
constexpr bool upper_matrix = true;
using fp_matrix = typename fw::blas::triangular_matrix<real_t, L, fw::blas::triangular_matrix_type::upper, BM, BE>;
#else
constexpr bool upper_matrix = false;
using fp_matrix = typename fw::blas::triangular_matrix<real_t, L, fw::blas::triangular_matrix_type::lower, BM, BE>;
#endif

void blas_triangular_solve(const bool transpose, const std::size_t n, const real_t alpha, const std::vector<real_t>& a, std::vector<real_t>& x, const std::vector<real_t>& y);

void fp_triangular_solve(const bool transpose, const real_t alpha, const fp_matrix& a, std::vector<real_t>& x, std::vector<real_t>& y);

#endif
