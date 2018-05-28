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

#if defined(old)

constexpr std::size_t m_default = 256;
constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

constexpr std::size_t warmup = 100;
constexpr std::size_t measurement = 1000;

//TODO: create stream of matrices and then use compression

int main(int argc, char** argv)
{
        // read command line arguments
        const std::size_t m = (argc > 1 ? atoi(argv[1]) : m_default);
        const std::size_t n = (argc > 2 ? atoi(argv[2]) : n_default);
        const std::size_t num_matrices = (argc > 3 ? atoi(argv[3]) : num_matrices_default);
        const std::size_t bs = (argc > 4 ? atoi(argv[4]) : bs_default);
        std::cout << "matrix multiply: " << m << " x " << n << std::endl;
        std::cout << "num matrices: " << num_matrices << std::endl;
        
        // setup the matrix and vector
        std::vector<real_t> a(0), x(0), y_ref(0);
        a.reserve(num_matrices * m * n);
        x.reserve(num_matrices * n);
        y_ref.reserve(num_matrices * m);
        
        srand48(1);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                for (std::size_t j = 0; j < m; ++j)
                {
                        for (std::size_t i = 0; i < n; ++i)
                        {
                                a[k * m * n + j * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                        }
                }

                for (std::size_t i = 0; i < n; ++i)
                {
                        x[k * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                }
        }

        // reference
        const real_t alpha = static_cast<real_t>(1.0);
        const real_t beta = static_cast<real_t>(0.0);
        for (std::size_t k = 0; k < std::min(num_matrices, warmup + measurement); ++k)
        {
                fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y_ref[k * m], 1);
        }

        double max_gflops = 0.0;
        double max_abs_rel_error = 0.0;
        double y_error, y_ref_error;

        #pragma omp parallel
        {
                std::vector<real_t> y(0);
                y.reserve(num_matrices * m);

                // matrix vector multiply
                double time;
                if (BE == fw::fp<real_t>::default_bits_exponent() && BM == fw::fp<real_t>::default_bits_mantissa())
                {
                        std::cout << "compression: no" << std::endl;
                        std::cout << "matrix memory footprint: " << num_matrices * m * n * sizeof(real_t) / (1024 * 1024) << " MiB" << std::endl;
                        #pragma omp barrier

                        std::size_t k = 0; 
                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                k = (k + 1) % num_matrices;
                        }
                        
                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                k = (k + 1) % num_matrices;
                        }
                        time = omp_get_wtime() - time;
                }
                else
                {
                        std::cout << "compression: yes" << std::endl;
                        std::cout << "block size: " << bs << std::endl;
                        std::vector<real_t> buffer(0);

                        // create compressed matrix
                        using full_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::full>;
                        const std::size_t a_compressed_num_elements = full_matrix::num_elements<real_t, BE, BM>(m, n, bs);
                        using fp_t = typename fw::fp<real_t>::format<BE, BM>::type;
                        std::vector<fp_t> a_compressed(0);
                        a_compressed.reserve(num_matrices * a_compressed_num_elements);
                        std::cout << "matrix memory footprint: " << num_matrices * a_compressed_num_elements * sizeof(fp_t) / (1024 * 1024) << " MiB" << std::endl;
                        for (std::size_t k = 0; k < num_matrices; ++k)
                        {
                                full_matrix::compress<real_t, BE, BM>(m, n, &a[k * m * n], n, &a_compressed[k * a_compressed_num_elements], bs, &buffer);
                        }

                        #pragma omp barrier

                        std::size_t k = 0;
                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
                                k = (k + 1) % num_matrices;
                        }

                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
                                k = (k + 1) % num_matrices;
                        }
                        time = omp_get_wtime() - time;
                }

                #pragma omp barrier

                // correctness?
                double err = 0.0;
                double tmp_1, tmp_2, tmp_3;
                for (std::size_t k = 0; k < std::min(num_matrices, warmup + measurement); ++k)
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
                        double gflops = m * (2 * n - 1) / (time / measurement) * 1.0E-9;
                        max_gflops = std::max(max_gflops, gflops);
                        if (err > max_abs_rel_error)
                        {
                                max_abs_rel_error = std::max(max_abs_rel_error, err);
                                y_ref_error = tmp_2;
                                y_error = tmp_3;
                        }
                }
        }

        std::cout << "gflops: " << max_gflops << std::endl;
        std::cout << "max abs error: " << max_abs_rel_error << " (" << y_ref_error << " vs. " << y_error << ")" << std::endl;

        return 0;
}

#else

constexpr std::size_t m_default = 256;
constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

constexpr std::size_t warmup = 1;
constexpr std::size_t measurement = 1;

int main(int argc, char** argv)
{
        // read command line arguments
        const std::size_t m = (argc > 1 ? atoi(argv[1]) : m_default);
        const std::size_t n = (argc > 2 ? atoi(argv[2]) : n_default);
        const std::size_t num_matrices = (argc > 3 ? atoi(argv[3]) : num_matrices_default);
        const std::size_t bs = (argc > 4 ? atoi(argv[4]) : bs_default);
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
                for (std::size_t j = 0; j < m; ++j)
                {
                        for (std::size_t i = 0; i < n; ++i)
                        {
                                a[k * m * n + j * n + i] = static_cast<real_t>(2.0 * drand48() - 1.0);
                        }
                }

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
        using full_matrix = fw::blas::blocked_matrix<fw::blas::matrix_type::full>;
        const std::size_t a_compressed_num_elements = full_matrix::num_elements<real_t, BE, BM>(m, n, bs);
        using fp_t = typename fw::fp<real_t>::format<BE, BM>::type;
        std::vector<fp_t> a_compressed(0);
        a_compressed.reserve(num_matrices * a_compressed_num_elements);
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
                full_matrix::compress<real_t, BE, BM>(m, n, &a[k * m * n], n, &a_compressed[k * a_compressed_num_elements], bs);
        }

        double max_gflops = 0.0;
        double max_abs_rel_error = 0.0;
        double y_error, y_ref_error;

        #pragma omp parallel
        {
                // matrix vector multiply
                double time;
                if (BE == fw::fp<real_t>::default_bits_exponent() && BM == fw::fp<real_t>::default_bits_mantissa())
                {
                        std::cout << "compression: no" << std::endl;
                        std::cout << "matrix memory footprint: " << num_matrices * m * n * sizeof(real_t) / (1024 * 1024) << " MiB" << std::endl;

                        #pragma omp barrier

                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {                       
                                        fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                }
                        }
                        
                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        fw::blas::gemv<real_t>(CblasRowMajor, CblasNoTrans, m, n, alpha, &a[k * m * n], n, &x[k * n], 1, beta, &y[k * m], 1);
                                }
                        }
                        time = omp_get_wtime() - time;
                }
                else
                {
                        std::cout << "compression: yes" << std::endl;
                        std::cout << "matrix memory footprint: " << num_matrices * a_compressed_num_elements * sizeof(fp_t) / (1024 * 1024) << " MiB" << std::endl;
                        std::cout << "block size: " << bs << std::endl;
                        std::vector<real_t> buffer(bs * bs);

                        #pragma omp barrier

                        for (std::size_t i = 0; i < warmup; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
                                }
                        }

                        time = omp_get_wtime();
                        for (std::size_t i = 0; i < measurement; ++i)
                        {
                                #pragma omp for
                                for (std::size_t k = 0; k < num_matrices; ++k)
                                {
                                        full_matrix_vector(false, m, n, alpha, &a_compressed[k * a_compressed_num_elements], &x[k * n], beta, &y[k * m], bs, &buffer);
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
                        double gflops = m * (2 * n - 1) / (time / measurement) * 1.0E-9;
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

#endif
