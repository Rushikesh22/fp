// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <matrix_vector_kernel.hpp>

constexpr std::size_t m_default = 256;
constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

constexpr std::size_t warmup = 5;
constexpr std::size_t measurement = 10;

int main(int argc, char** argv)
{
        // read command line arguments
        const std::size_t m = (argc > 1 ? atoi(argv[1]) : m_default);
        const std::size_t n = (argc > 2 ? atoi(argv[2]) : n_default);
        const std::size_t num_matrices = (argc > 3 ? atoi(argv[3]) : num_matrices_default);
        const std::size_t bs = (argc > 4 ? atoi(argv[4]) : bs_default);

        #if !defined(FULL_MATRIX)
        if (m != n)
        {
                std::cerr << "triangular matrix: m != n" << std::endl;
                return 1;
        }
        #endif

        std::cout << "matrix multiply: " << m << " x " << n << std::endl;
        std::cout << "num matrices: " << num_matrices << std::endl;

        // setup the matrix and vector
        std::vector<real_t> a(0), x(0), y_ref(0), y(0);
        a.reserve(num_matrices * m * n);
        x.reserve(num_matrices * n);
        y_ref.reserve(num_matrices * m);
        y.reserve(num_matrices * m);
        
        srand48(1);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                #if defined(FULL_MATRIX)
                for (std::size_t j = 0; j < m; ++j)
                {
                        for (std::size_t i = 0; i < n; ++i)
                        {
                                a[k * m * n + j * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                        }
                }
                #else
                for (std::size_t j = 0; j < n; ++j)
                {
                        for (std::size_t i = 0; i < j; ++i)
                        {
                                a[k * n * n + j * n + i] = static_cast<real_t>(0.0);
                        }
  
                        for (std::size_t i = j; i < n; ++i)
                        {
                                a[k * n * n + j * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                        }
                }
                #endif

                for (std::size_t i = 0; i < n; ++i)
                {
                        x[k * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                }
        }

        // reference
        const real_t alpha = static_cast<real_t>(1.0);
        const real_t beta = static_cast<real_t>(0.0);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y_ref[k * m], 1);
        }

        // create compressed matrix
        std::vector<fp_t> a_compressed(0);
        #if defined(FULL_MATRIX)
        using full_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::full>;
        const std::size_t a_compressed_num_elements = full_matrix::num_elements<real_t, BE, BM>(m, n, bs);
        using fp_t = typename fw::fp<real_t>::format<BE, BM>::type;
        a_compressed.reserve(num_matrices * a_compressed_num_elements);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                full_matrix::compress<real_t, BE, BM>(m, n, &a[k * m * n], n, &a_compressed[k * a_compressed_num_elements], bs);
        }
        #else
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
                        upper_matrix::compress<real_t, BE, BM>(n, &a[k * m * n], n, &a_compressed[k * a_compressed_num_elements], bs);
                }
        }
        #endif

        double max_gflops = 0.0;
        double max_abs_rel_error = 0.0;
        double y_error, y_ref_error;

        #pragma omp parallel
        {
                // matrix vector multiply
                double time;
                if (BE == fw::fp<real_t>::default_bits_exponent() && BM == fw::fp<real_t>::default_bits_mantissa())
                {
                        #pragma omp master
                        {
                                std::cout << "compression: no" << std::endl;
                                std::cout << "matrix memory footprint: " << num_matrices * m * n * sizeof(real_t) / (1024 * 1024) << " MiB" << std::endl;
                        }

                        #pragma omp barrier

                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {                       
                                        #if defined(FULL_MATRIX)
                                        fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                        #else
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                y[k * n + kk] = x[k * n + kk];
                                        }
                                        fw::blas::tpmv<real_t>(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, reinterpret_cast<const real_t*>(&a_compressed[k * a_compressed_num_elements]), &y[k * n], 1);
                                        #endif
                                }
                        }
                        
                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        #if defined(FULL_MATRIX)
                                        fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                        #else
                                        for (std::size_t kk = 0; kk < n; ++kk)
                                        {
                                                y[k * n + kk] = x[k * n + kk];
                                        }
                                        fw::blas::tpmv<real_t>(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, reinterpret_cast<const real_t*>(&a_compressed[k * a_compressed_num_elements]), &y[k * n], 1);
                                        #endif
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
                                        #if defined(FULL_MATRIX)
                                        full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
                                        #else
                                        upper_triangle_matrix_vector(false, n, &a_compressed[k * a_compressed_num_elements], &x[k * n], &y[k * m], bs, &buffer);
                                        #endif
                                }
                        }

                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        #if defined(FULL_MATRIX)
                                        full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
                                        #else
                                        upper_triangle_matrix_vector(false, n, &a_compressed[k * a_compressed_num_elements], &x[k * n], &y[k * m], bs, &buffer);
                                        #endif
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
                        for (std::size_t i = 0; i < m; ++i)
                        {
                                const double tmp_1 = std::abs((y_ref[k * m + i] - y[k * m + i]) / y_ref[k * m + i]);
                                if (tmp_1 > err)
                                {
                                        err = std::max(err, tmp_1);
                                        tmp_2 = y_ref[k * m + i];
                                        tmp_3 = y[k * m + i];
                                }
                        }
                }

                #pragma omp critical
                {
                        #if defined(FULL_MATRIX)
                        double gflops = m * (2 * n - 1) / (time / measurement) * 1.0E-9;
                        #else
                        double gflops = (n * (2 * n - 1) / 2) / (time / measurement) * 1.0E-9;
                        #endif
                        max_gflops = std::max(max_gflops, gflops);
                        if (err > max_abs_rel_error)
                        {
                                max_abs_rel_error = std::max(max_abs_rel_error, err);
                                y_ref_error = tmp_2;
                                y_error = tmp_3;
                        }
                }
        }

        std::cout << "gflops: " << max_gflops * num_matrices << std::endl;
        std::cout << "max abs error: " << max_abs_rel_error << " (" << y_ref_error << " vs. " << y_error << ")" << std::endl;

        return 0;
}
