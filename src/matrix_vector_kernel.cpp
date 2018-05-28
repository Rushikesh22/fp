// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include "matrix_vector_kernel.hpp"

void full_matrix_vector(const bool transpose, const std::size_t m, const std::size_t n, const real_t alpha, const fp_t* a, const real_t* x, const real_t beta, real_t* y, const std::size_t bs, std::vector<real_t>* buffer)
{
        using full_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::full>;
        full_matrix::pmv<real_t, BE, BM>(transpose, bs, m, n, alpha, a, x, beta, y, buffer);
        //fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[0], n, &x[0], 1, beta, &y[0], 1);
}