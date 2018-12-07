// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(GENERAL_MATRIX_VECTOR_KERNEL_HPP)
#define GENERAL_MATRIX_VECTOR_KERNEL_HPP

#include <cstdint>
#include <vector>
#include <fp/fp_blas.hpp>

// fundamental real data type: 'float' or 'double'
//using real_t = float;
using real_t = double;
using vec_t = double;
using mat_t = float;

// number of bits to be used for exponent and mantissa
#if defined(_BE)
static constexpr std::uint32_t BE = _BE;
#else
static constexpr std::uint32_t BE = 0;
#endif

#if defined(_BM)
static constexpr std::uint32_t BM = _BM;
#else
static constexpr std::uint32_t BM = 16;
#endif

#if defined(_ROWMAJOR)
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::rowmajor;
#elif defined(_COLMAJOR)
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::colmajor;
#else
static constexpr fw::blas::matrix_layout L = fw::blas::matrix_layout::rowmajor;
#endif

constexpr CBLAS_LAYOUT layout = (L == fw::blas::matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

// compressed matrix data type
using fp_matrix = typename fw::blas::matrix<real_t, L, BM, BE>;

// prototypes
#include "general_matrix_vector_kernel_blas.hpp"

double fp_matrix_vector(const bool transpose, const mat_t alpha, const fp_matrix& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y);

#endif
