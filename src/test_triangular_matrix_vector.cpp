// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <triangular_matrix_vector_kernel.hpp>

#if defined(THREAD_PINNING)
#include <sched.h>
#include <sys/sysinfo.h>
#endif

constexpr std::size_t m_default = 256;
constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

#if defined(BENCHMARK)
constexpr std::size_t warmup = 5;
constexpr std::size_t measurement = 10;
#else
constexpr std::size_t warmup = 0;
constexpr std::size_t measurement = 1;
#endif

void kernel(const real_t alpha, const real_t beta, const bool transpose,
    const std::size_t n,
    const std::vector<std::vector<real_t>>& a,
    const std::vector<fp_matrix>& a_compressed,
    const std::vector<std::vector<real_t>>& x,
    std::vector<std::vector<real_t>>& y_ref,
    std::vector<std::vector<real_t>>& y,
    const bool symmetric = false,
    const bool use_blas = false);

int main(int argc, char** argv)
{
    // read command line arguments
    const std::size_t n = (argc > 1 ? atoi(argv[1]) : n_default);
    const std::size_t num_matrices = (argc > 2 ? atoi(argv[2]) : num_matrices_default);
    const std::size_t bs = (argc > 3 ? atoi(argv[3]) : bs_default);
    const bool symmetric = (argc > 4 ? (atoi(argv[4]) != 0 ? true : false) : false);
    const bool use_blas = (argc > 5 ? (atoi(argv[5]) != 0 ? true : false) : false);

    std::cout << "triangular matrix multiply: " << n << " x " << n << " (" << (upper_matrix ? "upper)" : "lower)") << std::endl;
    std::cout << "num matrices: " << num_matrices << std::endl;
    std::cout << "block size: " << bs << std::endl;

    #if defined(THREAD_PINNING)
    #pragma omp parallel
    {
        const std::size_t thread_id = omp_get_thread_num();
        const std::size_t num_cpus = get_nprocs_conf();

        cpu_set_t cpu_mask;
        CPU_ZERO(&cpu_mask);
        CPU_SET(thread_id % num_cpus, &cpu_mask);
        sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask);
    }
    #endif

    // create matrices and vectors
    std::vector<std::vector<real_t>> a(num_matrices), x(num_matrices), y_ref(num_matrices), y(num_matrices);
    std::vector<fp_matrix> a_compressed;
    a_compressed.reserve(num_matrices);
    for (std::size_t k = 0; k < num_matrices; ++k)
    {
        a[k].reserve(n * n);
        srand48(k + 1);
        for (std::size_t j = 0; j < n; ++j)
        {
            if (upper_matrix)
            {
                for (std::size_t i = 0; i < j; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, n, n)] = 0.0;
                }
                for (std::size_t i = j; i < n; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, n, n)] = 0.9 + 0.2 * drand48();
                }
            }
            else
            {
                for (std::size_t i = 0; i <= j; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, n, n)] = 0.9 + 0.2 * drand48();
                }
                for (std::size_t i = (j + 1); i < n; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, n, n)] = 0.0;
                }
            }
        }

        if (symmetric)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                const std::size_t i_start = (upper_matrix ? 0 : j + 1);
                const std::size_t i_end = (upper_matrix ? j : n);

                for (std::size_t i = i_start; i < i_end; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, n, n)] = a[k][fw::blas::idx<L>(i, j, n, n)];
                }
            }
        }

        x[k].reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[k][i] = 0.9 + 0.2 * drand48();
        }

        std::array<std::size_t, 1> extent({n});
        a_compressed.emplace_back(a[k], extent, n, bs);

        y_ref[k].reserve(n);
        y[k].reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            y_ref[k][i] = 0.0;
            y[k][i] = 0.0;
        }
        
    }

    #if defined(BENCHMARK)
    // parameters for the matrix vector multiplication
    const real_t alpha = static_cast<real_t>(1.0);
    const real_t beta = static_cast<real_t>(0.0);
    const bool transpose = false;
    kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
    #else
    {
        const real_t alpha = static_cast<real_t>(1.0);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(-1.1);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.0);
        const real_t beta = static_cast<real_t>(-0.5);
        const bool transpose = false;
        kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.0);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.34);
        const real_t beta = static_cast<real_t>(-1.343);
        const bool transpose = false;
        kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, n, a, a_compressed, x, y_ref, y, symmetric, use_blas);
        }    
    }
    #endif
    
    return 0;
}

void kernel(const real_t alpha, const real_t beta, const bool transpose,
    const std::size_t n,
    const std::vector<std::vector<real_t>>& a,
    const std::vector<fp_matrix>& a_compressed,
    const std::vector<std::vector<real_t>>& x,
    std::vector<std::vector<real_t>>& y_ref,
    std::vector<std::vector<real_t>>& y,
    const bool symmetric,
    const bool use_blas)
{
    // print some information
    std::cout << "alpha: " << alpha << ", beta: " << beta << ", transpose: " << (transpose ? "true" : "false") << ", symmetric: " << (symmetric ? "yes" : "no") << std::endl;

    // reference computation
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        fw::blas::gemv(layout, (transpose ? CblasTrans : CblasNoTrans), n, n, alpha, &a[k][0], n, &x[k][0], 1, beta, &y_ref[k][0], 1);
    }

    if (use_blas)
    {
        std::cout << "mode: standard blas (matrix memory consumption: " << a.size() * ((n * (n + 1)) / 2) * sizeof(real_t) / (1024 * 1024) << " MiB)" << std::endl;
    }
    else
    {
        std::cout << "mode: fp_matrix, BE = " << BE << ", BM = " << BM << " (matrix memory consumption: " << a.size() * a_compressed[0].memory_footprint_bytes() / (1024 * 1024) << " MiB)" << std::endl;
    }

    // create packed matrix
    std::vector<std::vector<real_t>> a_packed(a.size());
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        a_packed[k].reserve((n * (n + 1)) / 2);

        if (upper_matrix)
        {
            for (std::size_t j = 0, l = 0; j < n; ++j)
            {
                const std::size_t i_start = (L == fw::blas::matrix_layout::rowmajor ? j : 0);
                const std::size_t i_end = (L == fw::blas::matrix_layout::rowmajor ? n : (j + 1));
                for (std::size_t i = i_start; i < i_end; ++i, ++l)
                {
                    a_packed[k][l] = a[k][j * n + i];
                }
            }
        }
        else
        {
            for (std::size_t j = 0, l = 0; j < n; ++j)
            {
                const std::size_t i_start = (L == fw::blas::matrix_layout::rowmajor ? 0 : j);
                const std::size_t i_end = (L == fw::blas::matrix_layout::rowmajor ? (j + 1) : n);
                for (std::size_t i = i_start; i < i_end; ++i, ++l)
                {
                    a_packed[k][l] = a[k][j * n + i];
                }
            }
        }
    }

    // own implementation
    double time_start = 0.0;
    double time_stop = 0.0;

    #pragma omp parallel
    {
        for (std::size_t l = 0; l < warmup; ++l)
        {
            if (use_blas)
            {
                #pragma omp for
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    blas_triangular_matrix_vector(transpose, n, alpha, a_packed[k], x[k], beta, y[k], symmetric);
                }
            }
            else
            {
                #pragma omp for
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    fp_triangular_matrix_vector(transpose, alpha, a_compressed[k], x[k], beta, y[k], symmetric);
                }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            time_start = omp_get_wtime();
        }
        #pragma omp barrier

        //time_start = omp_get_wtime();
        for (std::size_t l = 0; l < measurement; ++l)
        {
            if (use_blas)
            {
                #pragma omp for
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    blas_triangular_matrix_vector(transpose, n, alpha, a_packed[k], x[k], beta, y[k], symmetric);
                }
            }
            else
            {
                #pragma omp for
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    fp_triangular_matrix_vector(transpose, alpha, a_compressed[k], x[k], beta, y[k], symmetric);
                }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            time_stop = omp_get_wtime();
        }
    }

    #if defined(BENCHMARK)
    // output some metrics
    const double gflops = measurement * a.size() * (symmetric ? 2 : 1) * (n * n) / (time_stop - time_start) * 1.0E-9;
    std::cout << "gflops: " << gflops << std::endl;
    #else
    // correctness
    double dev = 0.0;
    real_t v_1 = y_ref[0][0];
    real_t v_2 = y[0][0];
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const double tmp = std::abs((y[k][i] - y_ref[k][i]) / y_ref[k][i]);
            if (tmp > dev)
            {
                dev = tmp;
                v_1 = y_ref[k][i];
                v_2 = y[k][i];
            }
        }
    }
    std::cout << "deviation: " << dev << " (" << v_1 << " vs. " << v_2 << ")" << std::endl;
    #endif
}
