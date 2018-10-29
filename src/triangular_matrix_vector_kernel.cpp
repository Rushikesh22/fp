// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <triangular_matrix_vector_kernel.hpp>

void blas_triangular_matrix_vector(const bool transpose, const std::size_t n, const real_t alpha, const std::vector<real_t>& a, const std::vector<real_t>& x, const real_t beta, std::vector<real_t>& y, const bool symmetric)
{
    if (symmetric)
    {
        fw::blas::spmv(layout, (upper_matrix ? CblasUpper : CblasLower), n, alpha, &a[0], &x[0], 1, beta, &y[0], 1);
    }
    else
    {
        std::vector<real_t> buffer_y;
        buffer_y.reserve(n);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            buffer_y[i] = x[i];
        }

        fw::blas::tpmv(layout, (upper_matrix ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, n, &a[0], &buffer_y[0], 1);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] = alpha * buffer_y[i] + beta * y[i];
        }
    }
}

void fp_triangular_matrix_vector(const bool transpose, const real_t alpha, const fp_matrix& a, const std::vector<real_t>& x, const real_t beta, std::vector<real_t>& y, const bool symmetric)
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