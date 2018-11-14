// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(INTEGER_BLAS_HPP)
#define INTEGER_BLAS_HPP

#include <cstdint>

#if !defined(FP_NAMESPACE)
    #define FP_NAMESPACE fw
#endif

namespace FP_NAMESPACE
{
    namespace blas
    {
    #if defined(FP_INTEGER_GEMV)
        template <typename T_1, typename T_2>
        static void gemv(const matrix_layout layout, const bool transpose, const std::size_t m, const std::size_t n, const T_1* a, const T_2* x, T_2* y)
        {
            
            if ((transpose && layout == matrix_layout::rowmajor) ||
                (!transpose && layout == matrix_layout::colmajor))
            {
                const std::size_t N = (transpose ? n : m);
                const std::size_t M = (transpose ? m : n);

                for (std::size_t j = 0; j < N; ++j)
                {
                    y[j] = 0;
                }

            #if defined(__AVX2__) || defined(__AVX512F__)
                constexpr bool use_simd_intrinsics = std::is_same<T_1, std::uint8_t>::value;
                if (use_simd_intrinsics)
                {
                    constexpr std::size_t chunk_size = 32;
                    constexpr std::size_t chunks = 4;
                    if (N < (chunks * chunk_size))
                    {
                        const std::size_t inc_i = (chunks * chunk_size + (N - 1)) / N;
                        alignas(32) std::int16_t buffer_a[inc_i * N];
                        for (std::size_t i = 0; i < M; i += inc_i)
                        {
                            const std::size_t ii_max = std::min(M - i, inc_i);
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[i * N], &buffer_a[0], ii_max * N);

                            for (std::size_t ii = 0; ii < ii_max; ++ii)
                            {
                                for (std::size_t j = 0; j < N; ++j)
                                {
                                    y[j] += buffer_a[ii * N + j] * x[i + ii];
                                }
                            }
                        }   
                    }
                    else
                    {
                        alignas(32) std::int16_t buffer_a[N];
                        for (std::size_t i = 0; i < M; ++i)
                        {
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[i * N], &buffer_a[0], N);

                            for (std::size_t j = 0; j < N; ++j)
                            {
                                y[j] += buffer_a[j] * x[i];
                            }
                        }   
                    }
                }
                else
            #endif
                {
                    for (std::size_t i = 0; i < M; ++i)
                    {
                        for (std::size_t j = 0; j < N; ++j)
                        {
                            y[j] += a[i * N + j] * x[i];
                        }
                    }
                }
            }
            else
            {
                const std::size_t N = (transpose ? n : m);
                const std::size_t M = (transpose ? m : n);

            #if defined(__AVX2__) || defined(__AVX512F__)
                constexpr bool use_simd_intrinsics = std::is_same<T_1, std::uint8_t>::value;
                if (use_simd_intrinsics)
                {
                    constexpr std::size_t chunk_size = 32;
                    constexpr std::size_t chunks = 4;
                    if (M < (chunks * chunk_size))
                    {
                        const std::size_t inc_j = (chunks * chunk_size + (M - 1)) / M;
                        alignas(32) std::int16_t buffer_a[inc_j * M];
                        for (std::size_t j = 0; j < N; j += inc_j)
                        {
                            const std::size_t jj_max = std::min(N - j, inc_j);
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], jj_max * M);

                            for (std::size_t jj = 0; jj < jj_max; ++jj)
                            {
                                T_2 tmp = 0;
                                for (std::size_t i = 0; i < M; ++i)
                                {
                                    tmp += buffer_a[jj * M + i] * x[i];
                                }
                                y[j + jj] = tmp;
                            }
                        }   
                    }
                    else
                    {
                        alignas(32) std::int16_t buffer_a[M];
                        for (std::size_t j = 0; j < N; ++j)
                        {
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], M);

                            T_2 tmp = 0;
                            for (std::size_t i = 0; i < M; ++i)
                            {
                                tmp += buffer_a[i] * x[i];
                            }
                            y[j] = tmp;
                        }
                    }
                }
                else
            #endif
                {
                    for (std::size_t j = 0; j < N; ++j)
                    {
                        T_2 tmp = 0;
                        for (std::size_t i = 0; i < M; ++i)
                        {
                            tmp += a[j * M + i] * x[i];
                        }
                        y[j] = tmp;
                    }
                }
            }
        }
    #endif
    }
}

#endif