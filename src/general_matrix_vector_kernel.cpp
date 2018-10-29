// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <general_matrix_vector_kernel.hpp>

void blas_matrix_vector(const bool transpose, const std::size_t m, const std::size_t n, const real_t alpha, const std::vector<real_t>& a, const std::vector<real_t>& x, const real_t beta, std::vector<real_t>& y)
{
    const std::size_t lda = (L == fw::blas::matrix_layout::rowmajor ? n : m);
    fw::blas::gemv(layout, (transpose ? CblasTrans : CblasNoTrans), m, n, alpha, &a[0], lda, &x[0], 1, beta, &y[0], 1);
}

void fp_matrix_vector(const bool transpose, const real_t alpha, const fp_matrix& a, const std::vector<real_t>& x, const real_t beta, std::vector<real_t>& y)
{
    a.matrix_vector(transpose, alpha, x, beta, y);
}