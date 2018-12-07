// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <triangular_solve_kernel.hpp>

#include "general_matrix_vector_kernel_blas.cpp"

void blas_triangular_solve(const bool transpose, const std::size_t n, const mat_t alpha, const std::vector<real_t>& a, std::vector<vec_t>& x, const std::vector<vec_t>& y)
{
    const mat_t* ptr_a = reinterpret_cast<const mat_t*>(&a[0]);
    std::vector<mat_t> tmp_a;
    if (!std::is_same<mat_t, real_t>::value)
    { 
        const std::size_t num_elements = (n * (n + 1)) / 2;
        tmp_a.reserve(num_elements);
        for (std::size_t i = 0; i < num_elements; ++i)
        {
            tmp_a[i] = a[i];
        }
        ptr_a = &tmp_a[0];
    }
    
    mat_t* ptr_x = reinterpret_cast<mat_t*>(&x[0]);
    std::vector<mat_t> tmp_x;
    if (!std::is_same<mat_t, vec_t>::value)
    {
            tmp_x.reserve(n);
            ptr_x = &tmp_x[0];
    }

    const real_t inv_alpha = 1.0 / alpha;
    for (std::size_t i = 0; i < n; ++i)
    {
            ptr_x[i] = y[i] * inv_alpha;
    }

    fw::blas::tpsv(layout, (upper_matrix ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, n, ptr_a, ptr_x, 1);

    if (!std::is_same<mat_t, vec_t>::value)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = ptr_x[i];
        }
    }
}

void fp_triangular_solve(const bool transpose, const mat_t alpha, const fp_matrix& a, std::vector<vec_t>& x, std::vector<vec_t>& y)
{
    a.triangular_solve(transpose, alpha, x, y);
}