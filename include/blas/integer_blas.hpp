// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(INTEGER_BLAS_HPP)
#define INTEGER_BLAS_HPP

#include <cstdint>
#include <fp/fp.hpp>

#if !defined(FP_NAMESPACE)
    #define FP_NAMESPACE fw
#endif

namespace FP_NAMESPACE
{
    namespace blas
    {
        using FP_NAMESPACE::internal::alignment;

        //! \brief General matrix vector multiplication with integer matrix and floating point vectors
        //!
        //! The integer matrix 'a' is assumed to be either of 8 or 16-bit integer type.
        //! Other than for the cblas_*gemv call, there is no 'alpha' and 'beta' parameters:
        //! they are 1.0 and 0.0 implicitly!
        //!
        //! \tparam T_1 integer input type of matrix 'a'
        //! \tparam T_2 floating point input and output type of vectors 'x' and 'y'
        //! \param layout row or column major order
        //! \param transpose transpose matrix 'a'
        //! \param m number of rows of matrix 'a'
        //! \param n number of columns of matrix 'a'
        //! \param a matrix
        //! \param x input vector
        //! \param y output vector
        template <typename T_1, typename T_2, typename X = typename std::enable_if<!(std::is_integral<T_1>::value && std::is_floating_point<T_2>::value)>::type>
        static void gemv(const matrix_layout layout, const bool transpose, const std::size_t m, const std::size_t n, const T_1* a, const T_2* x, T_2* y, const X* dummy = nullptr);

        template <typename T_1, typename T_2, typename X = typename std::enable_if<std::is_integral<T_1>::value && std::is_floating_point<T_2>::value>::type>
        static void gemv(const matrix_layout layout, const bool transpose, const std::size_t m, const std::size_t n, const T_1* a, const T_2* x, T_2* y)
        {
            static_assert(std::is_integral<T_1>::value && (8 * sizeof(T_1) <= 16) && std::is_floating_point<T_2>::value, "error: only integer matrix and floating point vectors are allowed");

            constexpr T_2 f_0 = static_cast<T_2>(0.0);
            constexpr T_2 f_1 = static_cast<T_2>(1.0);

            if (m == 0 || n == 0) return;

            if ((transpose && layout == matrix_layout::rowmajor) ||
                (!transpose && layout == matrix_layout::colmajor))
            {
                // matrix extent
                const std::size_t M = (transpose ? m : n);
                const std::size_t N = (transpose ? n : m);

                // set the output vector to zero: we will accumulate on it
                for (std::size_t j = 0; j < N; ++j)
                {
                    y[j] = f_0;
                }

            #if defined(__AVX2__) || defined(__AVX512F__)
                // use SIMD intrinsics only if the input matrix is of 8-bit integer type
                constexpr bool use_simd_intrinsics = std::is_same<T_1, std::uint8_t>::value;
                if (use_simd_intrinsics)
                {
                    // 32 8-bit integers fit into one AVX2 register
                    constexpr std::size_t chunk_size = 32;
                    // we need to operate some more numbers than 32 to get performance
                    constexpr std::size_t chunks = 4;
                    // if 'N' is too small, load multiple rows at once and operate on the respective tile
                    if (N < (chunks * chunk_size))
                    {
                        // determine the number of rows to be loaded
                        const std::size_t inc_i = (chunks * chunk_size + (N - 1)) / N;
                        alignas(alignment) std::int16_t buffer_a[inc_i * N];
                        // matrix vector multiplication on the current block
                        for (std::size_t i = 0; i < M; i += inc_i)
                        {
                            // unpack 8-bit into 16-bit integers: performance reasons
                            const std::size_t ii_max = std::min(M - i, inc_i);
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[i * N], &buffer_a[0], ii_max * N);

                            // matrix vector multiplication on the tile loaded
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
                        alignas(alignment) std::int16_t buffer_a[N];
                        // matrix vector multiplication on the current block
                        for (std::size_t i = 0; i < M; ++i)
                        {
                            // unpack 8-bit into 16-bit integers: performance reasons
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
                    // matrix vector multiplication on the current block
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
                // matrix extent
                const std::size_t M = (transpose ? m : n);
                const std::size_t N = (transpose ? n : m);                

                // set the output vector to zero: we will accumulate on it
                for (std::size_t j = 0; j < N; ++j)
                {
                    y[j] = f_0;
                }

            #if defined(__AVX2__) || defined(__AVX512F__)
                constexpr bool use_simd_intrinsics = std::is_same<T_1, std::uint8_t>::value;
                if (use_simd_intrinsics)
                {
                    // 32 8-bit integers fit into one AVX2 register
                    constexpr std::size_t chunk_size = 32;
                    // we need to operate some more numbers than 32 to get performance
                    constexpr std::size_t chunks = 4;
                    // if 'M' is too small, load multiple rows at once and operate on the respective tile
                    if (M < (chunks * chunk_size))
                    {
                        // determine the number of rows to be loaded
                        const std::size_t inc_j = (chunks * chunk_size + (M - 1)) / M;
                        alignas(alignment) std::int16_t buffer_a[inc_j * M];
                        // matrix vector multiplication on the current block
                        for (std::size_t j = 0; j < N; j += inc_j)
                        {
                            // unpack 8-bit into 16-bit integers: performance reasons
                            const std::size_t jj_max = std::min(N - j, inc_j);
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], jj_max * M);

                            // matrix vector multiplication on the tile loaded
                            for (std::size_t jj = 0; jj < jj_max; ++jj)
                            {
                                T_2 tmp = f_0;
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
                        alignas(alignment) std::int16_t buffer_a[M];
                        // matrix vector multiplication on the current block
                        for (std::size_t j = 0; j < N; ++j)
                        {
                            // unpack 8-bit into 16-bit integers: performance reasons
                            internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], M);

                            T_2 tmp = f_0;
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
                    // matrix vector multiplication on the current block
                    for (std::size_t j = 0; j < N; ++j)
                    {
                        T_2 tmp = f_0;
                        for (std::size_t i = 0; i < M; ++i)
                        {
                            tmp += a[j * M + i] * x[i];
                        }
                        y[j] = tmp;
                    }
                }
            }
        }
        
        //! \brief General matrix vector multiplication with integer matrix and floating point vectors
        //!
        //! This matrix applies matrix 'a' to 'x_1' and T('a') to 'x_2' and writes the output to 'y_1' and 'y_2'.
        //!
        //! The integer matrix 'a' is assumed to be either of 8 or 16-bit integer type.
        //! Other than for the cblas_*gemv call, there is no 'alpha' and 'beta' parameters:
        //! they are 1.0 and 0.0 implicitly!
        //!
        //! \tparam T_1 integer input type of matrix 'a'
        //! \tparam T_2 floating point input and output type of vectors 'x' and 'y'
        //! \param layout row or column major order
        //! \param transpose transpose matrix 'a'
        //! \param m number of rows of matrix 'a'
        //! \param n number of columns of matrix 'a'
        //! \param a matrix
        //! \param x_1 input vector 1
        //! \param y_1 output vector 1
        //! \param x_2 input vector 2
        //! \param y_2 output vector 2
        template <typename T_1, typename T_2, typename X = typename std::enable_if<!(std::is_integral<T_1>::value && std::is_floating_point<T_2>::value)>::type>
        static void gem2v(const matrix_layout layout, const std::size_t m, const std::size_t n, const T_1* a, const T_2* x_1, T_2* y_1, const T_2* x_2, T_2* y_2, const X* dummy = nullptr);

        template <typename T_1, typename T_2, typename X = typename std::enable_if<std::is_integral<T_1>::value && std::is_floating_point<T_2>::value>::type>
        static void gem2v(const matrix_layout layout, const std::size_t m, const std::size_t n, const T_1* a, const T_2* x_1, T_2* y_1, const T_2* x_2, T_2* y_2)
        {
            static_assert(std::is_integral<T_1>::value && (8 * sizeof(T_1) <= 16) && std::is_floating_point<T_2>::value, "error: only integer matrix and floating point vectors are allowed");

            constexpr T_2 f_0 = static_cast<T_2>(0.0);
            constexpr T_2 f_1 = static_cast<T_2>(1.0);

            if (m == 0 || n == 0) return;

            const T_2* ptr_x_1 = (layout == matrix_layout::rowmajor ? x_1 : x_2);
            const T_2* ptr_x_2 = (layout == matrix_layout::rowmajor ? x_2 : x_1);

            T_2* ptr_y_1 = (layout == matrix_layout::rowmajor ? y_1 : y_2);
            T_2* ptr_y_2 = (layout == matrix_layout::rowmajor ? y_2 : y_1);

            // matrix extent                    
            const std::size_t N = (layout == matrix_layout::rowmajor ? m : n);
            const std::size_t M = (layout == matrix_layout::rowmajor ? n : m);

            // set the output vectors to zero: we will accumulate on them
            for (std::size_t j = 0; j < N; ++j)
            {
                ptr_y_1[j] = f_0;
            }

            for (std::size_t j = 0; j < M; ++j)
            {
                ptr_y_2[j] = f_0;
            }

        #if defined(__AVX2__) || defined(__AVX512F__)
            // use SIMD intrinsics only if the input matrix is of 8-bit integer type
            constexpr bool use_simd_intrinsics = std::is_same<T_1, std::uint8_t>::value;
            if (use_simd_intrinsics)
            {
                // 32 8-bit integers fit into one AVX2 register
                constexpr std::size_t chunk_size = 32;
                // we need to operate some more numbers than 32 to get performance
                constexpr std::size_t chunks = 4;
                // if 'M' is too small, load multiple rows at once and operate on the respective tile
                if (M < (chunks * chunk_size))
                {
                    // determine the number of rows to be loaded
                    const std::size_t inc_j = (chunks * chunk_size + (M - 1)) / M;
                    alignas(alignment) std::int16_t buffer_a[inc_j * M];
                    // matrix vector multiplication on the current block
                    for (std::size_t j = 0; j < N; j += inc_j)
                    {
                        // unpack 8-bit into 16-bit integers: performance reasons
                        const std::size_t jj_max = std::min(N - j, inc_j);
                        internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], jj_max * M);

                        // matrix vector multiplication on the tile loaded
                        for (std::size_t jj = 0; jj < jj_max; ++jj)
                        {
                            T_2 tmp = f_0;
                            for (std::size_t i = 0; i < M; ++i)
                            {
                                const std::int32_t tmp_a = buffer_a[jj * M + i];
                                tmp += tmp_a * ptr_x_1[i];
                                ptr_y_2[i] += tmp_a * ptr_x_2[j + jj];
                            }
                            ptr_y_1[j + jj] = tmp;
                        }
                    }
                }
                else
                {
                    alignas(alignment) std::int16_t buffer_a[M];
                    // matrix vector multiplication on the current block
                    for (std::size_t j = 0; j < N; ++j)
                    {
                        // unpack 8-bit into 16-bit integers: performance reasons
                        internal::recode_simd_intrinsics<T_1, std::int16_t>(&a[j * M], &buffer_a[0], M);
                    
                        // matrix vector multiplication on the row loaded
                        T_2 tmp = f_0;
                        for (std::size_t i = 0; i < M; ++i)
                        {
                            const std::int32_t tmp_a = buffer_a[i];
                            tmp += tmp_a * ptr_x_1[i];
                            ptr_y_2[i] += tmp_a * ptr_x_2[j];
                        }
                        ptr_y_1[j] = tmp;
                    }
                }
            }
            else
        #endif
            {
                // matrix vector multiplication on the current block
                for (std::size_t j = 0; j < N; ++j)
                {
                    T_2 tmp = f_0;
                    for (std::size_t i = 0; i < M; ++i)
                    {
                        const std::int32_t tmp_a = a[j * M + i];
                        tmp += tmp_a * ptr_x_1[i];
                        ptr_y_2[i] += tmp_a * ptr_x_2[j];
                    }
                    ptr_y_1[j] = tmp;
                }
            }
        }
    }
}

#endif