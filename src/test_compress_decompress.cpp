// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <omp.h>
#if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
#include <triangular_matrix_vector_kernel.hpp>
#else
#include <general_matrix_vector_kernel.hpp>
#endif

#if defined(THREAD_PINNING)
#include <sched.h>
#include <sys/sysinfo.h>
#endif

using fp_type = typename fp_matrix::fp_type;

constexpr std::size_t m_default = 256;
constexpr std::size_t n_default = 256;
constexpr std::size_t num_matrices_default = 100;
constexpr std::size_t bs_default = 32;

#if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
using extent_t = std::array<std::size_t, 1>;
#else
using extent_t = std::array<std::size_t, 2>;
#endif

void kernel(const real_t alpha, const real_t beta, const bool transpose,
    const extent_t extent,
    const std::vector<std::vector<real_t>>& a,
    const std::vector<std::size_t>& lda,
    const std::vector<fp_type>& a_compressed,
    const std::size_t bs,
    const std::vector<std::vector<real_t>>& x,
    std::vector<std::vector<real_t>>& y_ref,
    std::vector<std::vector<real_t>>& y);

int main(int argc, char** argv)
{
    // read command line arguments
#if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
    const std::size_t n = (argc > 1 ? atoi(argv[1]) : n_default);
    const std::size_t num_matrices = (argc > 2 ? atoi(argv[2]) : num_matrices_default);
    const std::size_t bs = (argc > 3 ? atoi(argv[3]) : bs_default);
    std::cout << "triangular matrix multiply: " << n << " x " << n << " (" << (upper_matrix ? "upper)" : "lower)") << std::endl;
#else
    const std::size_t m = (argc > 1 ? atoi(argv[1]) : m_default);
    const std::size_t n = (argc > 2 ? atoi(argv[2]) : n_default);
    const std::size_t num_matrices = (argc > 3 ? atoi(argv[3]) : num_matrices_default);
    const std::size_t bs = (argc > 4 ? atoi(argv[4]) : bs_default);
    std::cout << "matrix multiply: " << m << " x " << n << std::endl;
#endif
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
    const std::size_t max_threads = omp_get_max_threads();
    std::vector<std::vector<real_t>> a(num_matrices), x(num_matrices), y_ref(num_matrices), y(num_matrices);
    std::vector<std::size_t> lda(num_matrices);

#if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
    const extent_t extent = std::array<std::size_t, 1>({n});
#else
    const extent_t extent = std::array<std::size_t, 2>({m, n});
#endif
    const std::size_t elements_fp_matrix = fp_matrix::memory_footprint_elements(extent, bs);
    std::vector<fp_type> a_compressed;
    a_compressed.reserve(num_matrices * elements_fp_matrix);

    #pragma omp parallel
    {
        const std::size_t thread_id = omp_get_thread_num();
        std::uint32_t seed = 1 + thread_id;
        
        #pragma omp for schedule(static)    
        for (std::size_t k = 0; k < num_matrices; ++k)
        {
        #if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
            const std::size_t ldn = n + n * 0.1 * rand_r(&seed) / RAND_MAX;
            a[k].reserve(ldn * n);
        #else
            const std::size_t ldn = (L == fw::blas::matrix_layout::rowmajor ? n : m) + n * 0.1 * rand_r(&seed) / RAND_MAX;
            a[k].reserve(ldn * (L == fw::blas::matrix_layout::rowmajor ? m : n));
        #endif
            lda[k] = ldn;

        #if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
            for (std::size_t j = 0; j < n; ++j)
            {
                if (upper_matrix)
                {
                    for (std::size_t i = 0; i < j; ++i)
                    {
                        a[k][fw::blas::idx<L>(j, i, ldn)] = 0.0;
                    }
                    for (std::size_t i = j; i < n; ++i)
                    {
                        a[k][fw::blas::idx<L>(j, i, ldn)] = 0.9 + 0.2 * rand_r(&seed) / RAND_MAX;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i <= j; ++i)
                    {
                        a[k][fw::blas::idx<L>(j, i, ldn)] = 0.9 + 0.2 * rand_r(&seed) / RAND_MAX;
                    }
                    for (std::size_t i = (j + 1); i < n; ++i)
                    {
                        a[k][fw::blas::idx<L>(j, i, ldn)] = 0.0;
                    }
                }
            }

            std::array<std::size_t, 1> extent({n});
            fp_matrix::compress(&a[k][0], ldn, &a_compressed[k * elements_fp_matrix], extent, bs);

            x[k].reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                x[k][i] = 0.9 + 0.2 * rand_r(&seed) / RAND_MAX;
            }

            y_ref[k].reserve(n);
            y[k].reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                y_ref[k][i] = 0.0;
                y[k][i] = 0.0;
            }
        #else
            for (std::size_t j = 0; j < m; ++j)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    a[k][fw::blas::idx<L>(j, i, ldn)] = 0.9 + 0.2 * rand_r(&seed) / RAND_MAX;
                }
            }

            std::array<std::size_t, 2> extent({m, n});
            fp_matrix::compress(&a[k][0], ldn, &a_compressed[k * elements_fp_matrix], extent, bs);

            const std::size_t mn = std::max(m, n);
            x[k].reserve(mn);
            for (std::size_t i = 0; i < mn; ++i)
            {
                x[k][i] = 0.9 + 0.2 * rand_r(&seed) / RAND_MAX;
            }

            y_ref[k].reserve(mn);
            y[k].reserve(mn);
            for (std::size_t i = 0; i < mn; ++i)
            {
                y_ref[k][i] = 0.0;
                y[k][i] = 0.0;
            }
        #endif
        }
    }

    {
        const real_t alpha = static_cast<real_t>(1.0);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(-1.1);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.0);
        const real_t beta = static_cast<real_t>(-0.5);
        const bool transpose = false;
        kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.0);
        const real_t beta = static_cast<real_t>(0.0);
        const bool transpose = false;
        kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        }
    }
    {
        const real_t alpha = static_cast<real_t>(0.34);
        const real_t beta = static_cast<real_t>(-2000.23);
        const bool transpose = false;
        kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        {
            const bool transpose = true;
            kernel(alpha, beta, transpose, extent, a, lda, a_compressed, bs, x, y_ref, y);
        }  
    }
    
    return 0;
}

void kernel(const real_t alpha, const real_t beta, const bool transpose,
    const extent_t extent,
    const std::vector<std::vector<real_t>>& a,
    const std::vector<std::size_t>& lda,
    const std::vector<fp_type>& a_compressed,
    const std::size_t bs,
    const std::vector<std::vector<real_t>>& x,
    std::vector<std::vector<real_t>>& y_ref,
    std::vector<std::vector<real_t>>& y)
{
    // print some information
    const std::size_t elements_fp_matrix = fp_matrix::memory_footprint_elements(extent, bs);
    std::cout << "alpha: " << alpha << ", beta: " << beta << ", transpose: " << (transpose ? "true" : "false") << std::endl;
    std::cout << "mode: fp_matrix, BE = " << BE << ", BM = " << BM << " (matrix memory consumption: " << a.size() * elements_fp_matrix * sizeof(fp_type) / (1024 * 1024) << " MiB)" << std::endl;

    #if defined(UPPER_MATRIX) || defined(LOWER_MATRIX)
    const std::size_t m = extent[0];
    const std::size_t n = m;
    #else
    const std::size_t m = extent[0];
    const std::size_t n = extent[1];
    #endif

    // reference computation
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        fw::blas::gemv(layout, (transpose ? CblasTrans : CblasNoTrans), m, n, alpha, &a[k][0], lda[k], &x[k][0], 1, beta, &y_ref[k][0], 1);
    }

    // decompress matrix
    #pragma omp parallel
    {
        std::vector<real_t> buffer;
        #pragma omp for
        for (std::size_t k = 0; k < a.size(); ++k)
        {
            const std::size_t num_elements = lda[k] * (L == fw::blas::matrix_layout::rowmajor ? m : n);
            buffer.reserve(num_elements);
            for (std::size_t i = 0; i < num_elements; ++i)
            {
                buffer[i] = 0.0;
            }
            fp_matrix::decompress(&a_compressed[k * elements_fp_matrix], &buffer[0], lda[k], extent, bs);
            fw::blas::gemv(layout, (transpose ? CblasTrans : CblasNoTrans), m, n, alpha, &buffer[0], lda[k], &x[k][0], 1, beta, &y[k][0], 1);
        }
    }

    // correctness
    double dev = 0.0;
    real_t v_1 = y_ref[0][0];
    real_t v_2 = y[0][0];
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        for (std::size_t j = 0; j < (transpose ? n : m); ++j)
        {
            const double tmp = std::abs((y[k][j] - y_ref[k][j]) / y_ref[k][j]);
            if (tmp > dev)
            {
                dev = tmp;
                v_1 = y_ref[k][j];
                v_2 = y[k][j];
            }
        }
    }
    std::cout << "deviation: " << dev << " (" << v_1 << " vs. " << v_2 << ")" << std::endl;
}
