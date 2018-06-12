// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <solve_kernel.hpp>

constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

constexpr std::size_t warmup = 5;
constexpr std::size_t measurement = 10;

int main(int argc, char** argv)
{
        // read command line arguments
        const std::size_t n = (argc > 1 ? atoi(argv[1]) : n_default);
        const std::size_t num_matrices = (argc > 2 ? atoi(argv[2]) : num_matrices_default);
        const std::size_t bs = (argc > 3 ? atoi(argv[3]) : bs_default);

        std::cout << "triangular solve: " << n << " x " << n << std::endl;
        std::cout << "num matrices: " << num_matrices << std::endl;

        // setup the matrix and vector
        std::vector<real_t> a(0), x_ref(0), x(0), tmp(0);
        a.reserve(num_matrices * n * n);
        x_ref.reserve(num_matrices * n);
        x.reserve(num_matrices * n);
        tmp.reserve(num_matrices * n);
        
        srand48(1);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {

                for (std::size_t j = 0; j < n; ++j)
                {
                        for (std::size_t i = 0; i < j; ++i)
                        {
                                a[k * n * n + j * n + i] = static_cast<real_t>(0.0);
                        }
  
                        for (std::size_t i = j; i < n; ++i)
                        {
                                a[k * n * n + j * n + i] = static_cast<real_t>(0.9 + 0.2 * drand48());
                        }
                }

                for (std::size_t i = 0; i < n; ++i)
                {
                        x_ref[k * n + i] = static_cast<real_t>(0.9 + 0.2 * drand48());
                }
        }

        // reference
        const real_t alpha = static_cast<real_t>(1.0);
        const real_t beta = static_cast<real_t>(0.0);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, n, n, alpha, &a[k * n * n], n, &x_ref[k * n], 1, beta, &tmp[k * n], 1);
        }

        // create compressed matrix
        std::vector<fp_t> a_compressed(0);
        using upper_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::upper>;
        const std::size_t a_compressed_num_elements = upper_matrix::num_elements<real_t, BE, BM>(n, bs);
        using fp_t = typename fw::fp<real_t>::format<BE, BM>::type;
        a_compressed.reserve(num_matrices * a_compressed_num_elements);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                if (BE == fw::fp<real_t>::default_bits_exponent() && BM == fw::fp<real_t>::default_bits_mantissa())
                {
                        for (std::size_t j = 0, kk = 0; j < n; ++j)
                        {
                                for (std::size_t i = j; i < n; ++i, ++kk)
                                {
                                        a_compressed[k * a_compressed_num_elements + kk] = a[k * n * n + j * n + i];
                                }
                        }
                }
                else
                {
                        upper_matrix::compress<real_t, BE, BM>(n, &a[k * n * n], n, &a_compressed[k * a_compressed_num_elements], bs);
                }
        }

        double max_gflops = 0.0;
        double max_abs_rel_error = 0.0;
        double x_error, x_ref_error;

        #pragma omp parallel
        {
                double time;
                if (BE == fw::fp<real_t>::default_bits_exponent() && BM == fw::fp<real_t>::default_bits_mantissa())
                {
                        #pragma omp master
                        {
                                std::cout << "compression: no" << std::endl;
                                std::cout << "matrix memory footprint: " << num_matrices * ((n * (n + 1)) / 2) * sizeof(real_t) / (1024 * 1024) << " MiB" << std::endl;
                        }

                        #pragma omp barrier

                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {                       
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                x[k * n + kk] = tmp[k * n + kk];
                                        }
                                        fw::blas::tpsv<real_t>(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, reinterpret_cast<const real_t*>(&a_compressed[k * a_compressed_num_elements]), &x[k * n], 1);
                                }
                        }
                        
                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                x[k * n + kk] = tmp[k * n + kk];
                                        }
                                        fw::blas::tpsv<real_t>(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, reinterpret_cast<const real_t*>(&a_compressed[k * a_compressed_num_elements]), &x[k * n], 1);
                                }
                        }
                        time = omp_get_wtime() - time;
                }
                else
                {
                        #pragma omp master
                        {
                                std::cout << "compression: yes" << std::endl;
                                std::cout << "matrix memory footprint: " << num_matrices * a_compressed_num_elements * sizeof(fp_t) / (1024 * 1024) << " MiB" << std::endl;
                                std::cout << "block size: " << bs << std::endl;
                        }
                        std::vector<real_t> buffer(bs * bs);

                        #pragma omp barrier

                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                x[k * n + kk] = tmp[k * n + kk];
                                        }
                                        triangular_solve(false, n, &a_compressed[k * a_compressed_num_elements], &x[k * n], bs, &buffer);
                                }
                        }

                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                x[k * n + kk] = tmp[k * n + kk];
                                        }
                                        triangular_solve(false, n, &a_compressed[k * a_compressed_num_elements], &x[k * n], bs, &buffer);
                                }
                        }
                        time = omp_get_wtime() - time;
                }

                #pragma omp barrier

                // correctness?
                double err = 0.0;
                double tmp_1, tmp_2, tmp_3;

                #pragma omp for
                for (std::size_t k = 0; k < num_matrices; ++k)
                {
                        for (std::size_t i = 0; i < n; ++i)
                        {
                                const double tmp_1 = std::abs((x_ref[k * n + i] - x[k * n + i]) / x_ref[k * n + i]);
                                if (tmp_1 > err)
                                {
                                        err = std::max(err, tmp_1);
                                        tmp_2 = x_ref[k * n + i];
                                        tmp_3 = x[k * n + i];
                                }
                        }
                }

                #pragma omp critical
                {
                        double gflops = (n * (2 * n - 1) / 2) / (time / measurement) * 1.0E-9;
                        max_gflops = std::max(max_gflops, gflops);
                        if (err > max_abs_rel_error)
                        {
                                max_abs_rel_error = std::max(max_abs_rel_error, err);
                                x_ref_error = tmp_2;
                                x_error = tmp_3;
                        }
                }
        }

        std::cout << "gflops: " << max_gflops * num_matrices << std::endl;
        std::cout << "max abs error: " << max_abs_rel_error << " (" << x_ref_error << " vs. " << x_error << ")" << std::endl;

        return 0;
}
