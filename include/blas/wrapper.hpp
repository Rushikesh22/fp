// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(WRAPPER_HPP)
#define WRAPPER_HPP

#include <cstdint>
#include <cblas.h>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw
#endif

namespace FP_NAMESPACE
{
    namespace blas
    {
        // BLAS call wrapper: matrix vector multiply
        template <typename T>
        static void gemv(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
            const std::size_t __M, const std::size_t __N, const T __alpha, const T* __A, const std::size_t __lda, 
            const T* __X, const std::size_t __incX, 
            const T __beta, T* __Y, const std::size_t __incY);
        
        template <>
        inline void gemv<double>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
            const std::size_t __M, const std::size_t __N, const double __alpha, const double* __A, const std::size_t __lda, 
            const double* __X, const std::size_t __incX, 
            const double __beta, double* __Y, const std::size_t __incY) 
        {
            cblas_dgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
        }

        template <>
        inline void gemv<float>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
            const std::size_t __M, const std::size_t __N, const float __alpha, const float* __A, const std::size_t __lda, 
            const float* __X, const std::size_t __incX, 
            const float __beta, float* __Y, const std::size_t __incY) 
        {
            cblas_sgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
        }

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
                            recode_simd_intrinsics<T_1, std::int16_t>(&a[i * N], &buffer_a[0], ii_max * N);

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
                            recode_simd_intrinsics<T_1, std::int16_t>(&a[i * N], &buffer_a[0], N);

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
                            recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], jj_max * M);

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
                            recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], M);

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

        // BLAS call wrapper: triangular packed matrix vector multiply
        template <typename T>
        static void tpmv(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const T* __Ap, T* __X, const std::size_t __incX);

        template <>
        inline void tpmv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const double* __Ap, double* __X, const std::size_t __incX)
        {
            cblas_dtpmv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
        }

        template <>
        inline void tpmv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const float* __Ap, float* __X, const std::size_t __incX)
        {
            cblas_stpmv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
        }

        // BLAS call wrapper: symmetric packed matrix vector multiply
        template <typename T>
        static void spmv(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo,
            const std::size_t __N, const T __alpha, const T* __Ap,
            const T* __X, const std::size_t __incX,
            const T __beta, T* __Y, const std::size_t __incY);

        template <>
        inline void spmv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo,
            const std::size_t __N, const double __alpha, const double* __Ap,
            const double* __X, const std::size_t __incX,
            const double __beta, double* __Y, const std::size_t __incY)
        {
            cblas_dspmv(__Order, __Uplo, __N, __alpha, __Ap, __X, __incX, __beta, __Y, __incY);
        }

        template <>
        inline void spmv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo,
            const std::size_t __N, const float __alpha, const float* __Ap,
            const float* __X, const std::size_t __incX,
            const float __beta, float* __Y, const std::size_t __incY)
        {
            cblas_sspmv(__Order, __Uplo, __N, __alpha, __Ap, __X, __incX, __beta, __Y, __incY);
        }

        // BLAS call wrapper: triangular packed solve
        template <typename T>
        static void tpsv(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const T* __Ap, T* __X, const std::size_t __incX);

        template <>
        inline void tpsv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const double* __Ap, double* __X, const std::size_t __incX)
        {
            cblas_dtpsv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
        }

        template <>
        inline void tpsv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
            const std::size_t __N, const float* __Ap, float* __X, const std::size_t __incX)
        {
            cblas_stpsv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
        }
    }
}

#endif
