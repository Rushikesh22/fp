// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include "solve_kernel.hpp"

void triangular_solve(const bool transpose, const std::size_t n, const fp_t* a, real_t* x, const std::size_t bs, std::vector<real_t>* buffer)
{
        using upper_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::upper>;
        upper_matrix::psv<real_t, BE, BM>(transpose, bs, n, a, x, buffer);
}