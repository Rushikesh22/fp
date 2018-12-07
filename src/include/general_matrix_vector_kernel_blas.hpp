// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(GENERAL_MATRIX_VECTOR_KERNEL_BLAS_HPP)
#define GENERAL_MATRIX_VECTOR_KERNEL_BLAS_HPP

double blas_matrix_vector(const bool transpose, const std::size_t m, const std::size_t n, const mat_t alpha, const std::vector<real_t>& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y);

#endif