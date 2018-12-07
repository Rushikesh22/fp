// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <general_matrix_vector_kernel.hpp>

void blas_matrix_vector(const bool transpose, const std::size_t m, const std::size_t n, const mat_t alpha, const std::vector<real_t>& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y)
{
    const std::size_t lda = (L == fw::blas::matrix_layout::rowmajor ? n : m);

    const mat_t* ptr_a = reinterpret_cast<const mat_t*>(&a[0]);
    std::vector<mat_t> tmp_a;
    if (!std::is_same<mat_t, real_t>::value)
    { 
        tmp_a.reserve(m * n);
        for (std::size_t i = 0; i < (m * n); ++i)
        {
            tmp_a[i] = a[i];
        }
        ptr_a = &tmp_a[0];
    }
    
    const mat_t* ptr_x = reinterpret_cast<const mat_t*>(&x[0]);
    mat_t* ptr_y = reinterpret_cast<mat_t*>(&y[0]);
    std::vector<mat_t> tmp_x, tmp_y;
    if (!std::is_same<mat_t, vec_t>::value)
    {
        tmp_x.reserve(transpose ? m : n);
        for (std::size_t i = 0; i < (transpose ? m : n); ++i)
        {
            tmp_x[i] = x[i];
        }
        ptr_x = &tmp_x[0];

        tmp_y.reserve(transpose ? n : m);
        for (std::size_t j = 0; j < (transpose ? n : m); ++j)
        {
            tmp_y[j] = y[j];
        }
        ptr_y = &tmp_y[0];
    }

    fw::blas::gemv(layout, (transpose ? CblasTrans : CblasNoTrans), m, n, alpha, ptr_a, lda, ptr_x, 1, static_cast<mat_t>(beta), ptr_y, 1);

    if (!std::is_same<mat_t, vec_t>::value)
    {
        for (std::size_t j = 0; j < (transpose ? n : m); ++j)
        {
            y[j] = tmp_y[j];
        }
    }
}

void fp_matrix_vector(const bool transpose, const mat_t alpha, const fp_matrix& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y)
{
    a.matrix_vector(transpose, alpha, x, beta, y);
}