// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <triangular_matrix_vector_kernel.hpp>
#include <omp.h>

#include "general_matrix_vector_kernel_blas.cpp"

double blas_triangular_matrix_vector(const bool transpose, const std::size_t n, const mat_t alpha, const std::vector<real_t>& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y, const bool symmetric)
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
    
    const mat_t* ptr_x = reinterpret_cast<const mat_t*>(&x[0]);
    mat_t* ptr_y = reinterpret_cast<mat_t*>(&y[0]);
    std::vector<mat_t> tmp_x, tmp_y;
    if (!std::is_same<mat_t, vec_t>::value)
    {
        tmp_x.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            tmp_x[i] = x[i];
        }
        ptr_x = &tmp_x[0];
    }

    double time = omp_get_wtime();
    {
        if (!std::is_same<mat_t, vec_t>::value || !symmetric)
        {
            tmp_y.reserve(n);
            if (symmetric)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    tmp_y[i] = y[i];
                }
            }
            else
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    tmp_y[i] = x[i];
                }
            }
            ptr_y = &tmp_y[0];
        }

        if (symmetric)
        {
            fw::blas::spmv(layout, (upper_matrix ? CblasUpper : CblasLower), n, alpha, ptr_a, ptr_x, 1, static_cast<mat_t>(beta), ptr_y, 1);

            if (!std::is_same<mat_t, vec_t>::value)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    y[i] = tmp_y[i];
                }
            }
        }
        else
        {
            fw::blas::tpmv(layout, (upper_matrix ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, n, ptr_a, ptr_y, 1);

            for (std::size_t i = 0; i < n; ++i)
            {
                y[i] = alpha * tmp_y[i] + beta * y[i];
            }
        }
    }
    time = (omp_get_wtime() - time);

    return time;
}

double fp_triangular_matrix_vector(const bool transpose, const mat_t alpha, const fp_matrix& a, const std::vector<vec_t>& x, const vec_t beta, std::vector<vec_t>& y, const bool symmetric)
{
    double time = omp_get_wtime();
    {
        if (symmetric)
        {
            a.symmetric_matrix_vector(alpha, x, beta, y);
        }
        else
        {
            a.matrix_vector(transpose, alpha, x, beta, y);
        }
    }
    return (omp_get_wtime() - time);
}