// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <triangular_solve_kernel.hpp>

void blas_triangular_solve(const bool transpose, const std::size_t n, const real_t alpha, const std::vector<real_t>& a, std::vector<real_t>& x, const std::vector<real_t>& y)
{
        const real_t inv_alpha = 1.0 / alpha;
        for (std::size_t i = 0; i < n; ++i)
        {
                x[i] = y[i] * inv_alpha;
        }

        fw::blas::tpsv(layout, (upper_matrix ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, n, &a[0], &x[0], 1);
}

void fp_triangular_solve(const bool transpose, const real_t alpha, const fp_matrix& a, std::vector<real_t>& x, std::vector<real_t>& y)
{
        a.solve(transpose, alpha, x, y);
}