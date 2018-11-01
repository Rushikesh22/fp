// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_BLAS_HPP)
#define FP_BLAS_HPP

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <array>
#include <cblas.h>
#include <omp.h>

#if defined(FP_MKL_INTEGER_GEMM_AVAILABLE)
#include <mkl.h>
#define FP_MAX_UINT16 0x7FFFU
#define FP_MAX_UINT8 0x7FU
#endif

#include <fp/fp.hpp>
#include <blas/wrapper.hpp>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw
#endif

#define FP_PREFETCH
#if defined(FP_PREFETCH) && !defined(FP_PREFETCH_LEVEL)
#define FP_PREFETCH_LEVEL 3
#endif

namespace FP_NAMESPACE
{
	namespace blas
	{
        //! default alignment: TODO replace by SIMD class alignment
        #if defined(__AVX512F__)
        constexpr std::size_t alignment = 64;
        #else
        constexpr std::size_t alignment = 32;
        #endif

        //! matrix format: colmajor, rowmajor
        enum class matrix_layout { colmajor = 0, rowmajor = 1 };

        // index computation
        template <matrix_layout L>
        inline std::size_t idx(const std::size_t j, const std::size_t i, const std::size_t m, const std::size_t n);

        template <>
        inline std::size_t idx<matrix_layout::rowmajor>(const std::size_t j, const std::size_t i, const std::size_t m, const std::size_t n)
        {
            return (j * n + i);
        }

        template <>
        inline std::size_t idx<matrix_layout::colmajor>(const std::size_t j, const std::size_t i, const std::size_t m, const std::size_t n)
        {
            return (i * m + j);
        }

        //! matrix types: lower, upper
        enum class triangular_matrix_type { lower = 0, upper = 1 };

        //! \brief General matrix
        //!
        //! \tparam T data type (of the matrix elements)
        //! \tparam BE number of bits in the exponent
        //! \tparam BM number of bits in the mantissa
        template <typename T, matrix_layout L = matrix_layout::rowmajor, std::uint32_t BE = fp<T>::default_bits_exponent(), std::uint32_t BM = fp<T>::default_bits_mantissa()>
        class matrix
        {
            static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

        public:

            // extent of the matrix: 'm' rows and 'n' columns
            const std::size_t m;
            const std::size_t n;

            // fp type for data compression
            using fp_type = typename FP_NAMESPACE::fp<T>::template format<BE, BM>::type;
            using fp_remapped_type = typename FP_NAMESPACE::internal::fp_remap<T, BE, BM>::type;

        private:

            static constexpr CBLAS_LAYOUT cblas_layout = (L == matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

        protected:

            // block size
            const std::size_t bs;

            // compressed matrix
            std::vector<fp_type> memory;
            const fp_type* compressed_data;

            const std::size_t num_blocks_a;
            const std::size_t num_elements_a;

            const std::size_t num_blocks_b;
            const std::size_t num_elements_b;

            const std::size_t num_blocks_c;
            const std::size_t num_elements_c;

            const std::size_t num_blocks_d;
            const std::size_t num_elements_d;

            const std::size_t num_elements;

            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // constructor that can be called by the triangular_matrix constructor
            template <typename TT>
            matrix(const TT* data, const std::array<std::size_t, 1>& extent, const std::size_t ld_data, triangular_matrix_type MT = triangular_matrix_type::upper, const std::size_t bs = bs_default)
                :
                m(extent[0]), // set, but not to be used
                n(extent[0]),
                bs(bs),
                compressed_data(nullptr),
                //  a b b | c
                //  0 a b | c
                //  0 0 a | c
                // -------+---
                //  0 0 0 | d
                num_blocks_a((n / bs)),
                num_elements_a(fp<T>::template format<BE, BM>::memory_footprint_elements((bs * (bs + 1)) / 2)),
                num_blocks_b((((n / bs) * ((n / bs) + 1)) / 2) - (n / bs)),
                num_elements_b(fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs)),
                num_blocks_c((n / bs) * (((n + bs - 1) / bs) - (n / bs))),
                num_elements_c(fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs))),
                num_blocks_d(((n + bs - 1) / bs) - (n / bs)),
                num_elements_d(fp<T>::template format<BE, BM>::memory_footprint_elements(((n - (n / bs) * bs) * (n - (n / bs) * bs + 1)) / 2)),
                num_elements(num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d)
            {
                ;
            }

            // template function providing a frame for different blas2 kernel implementations
            template <typename F>
            void blas2_frame(const F& kernel, const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (n == 0 || m == 0)
                {
                    return;
                }

                // handle some special cases
                if (alpha == f_0)
                {
                    if (beta == f_0)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            y[j] = f_0;
                        }
                    }
                    else if (beta != f_1)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            y[j] = beta * y[j];
                        }
                    }

                    return;
                }

                // allocate local memory
                const std::size_t mn = (transpose ? n : m);
                const bool use_buffer = (std::abs(y - x) >= std::max(m, n) ? false : true);
                alignas(alignment) T buffer_y[use_buffer ? mn : 0];
                T* ptr_y = nullptr;

                if (use_buffer)
                {
                    // accumulate on the buffer (do not write to the output directly)
                    ptr_y = &buffer_y[0];

                    // zero the buffer
                    #pragma omp simd
                    for (std::size_t j = 0; j < mn; ++j)
                    {
                        ptr_y[j] = f_0;
                    }
                }
                else
                {
                    // write to the output directly
                    ptr_y = y;

                    // scale by 'beta'
                    if (beta == f_0)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < mn; ++j)
                        {
                            ptr_y[j] = f_0;
                        }
                    }
                    else if (beta != f_1)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < mn; ++j)
                        {
                            ptr_y[j] *= beta;
                        }
                    }
                }

                // execute the kernel
                kernel(transpose, alpha, x, ptr_y);

                // output has been written directly
                if (!use_buffer)
                {
                    return;
                }

                // accumulate on 'y'
                if (beta == f_0)
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < mn; ++j)
                    {
                        y[j] = buffer_y[j];
                    }
                }
                else if (beta == f_1)
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < mn; ++j)
                    {
                        y[j] += buffer_y[j];
                    }
                }
                else
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < mn; ++j)
                    {
                        y[j] = buffer_y[j] + beta * y[j];
                    }
                }
            }

    public:

            // (default) block size
            static constexpr std::size_t bs_default = 32;

            // constructor
            matrix() = delete;

            template <typename TT>
            matrix(const TT* data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                m(extent[0]),
                n(extent[1]),
                bs(bs),
                compressed_data(nullptr),
                //  a a a | b
                //  a a a | b
                // -------+---
                //  c c c | d
                num_blocks_a((m / bs) * (n / bs)),
                num_elements_a(fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs)),
                num_blocks_b((m / bs) * (((n + bs - 1) / bs) - (n / bs))),
                num_elements_b(fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs))),
                num_blocks_c((((m + bs - 1) / bs) - (m / bs)) * (n / bs)),
                num_elements_c(fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * bs)),
                num_blocks_d((((m + bs - 1) / bs) - (m / bs)) * (((n + bs - 1) / bs) - (n / bs))),
                num_elements_d(fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * (n - (n / bs) * bs))),
                num_elements(num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d)
            {			
                if (std::is_same<TT, fp_type>::value)
                {
                    // 'data' points to a compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(data);
                }
                else
                {
                    // allocate memory for the compressed matrix
                    memory.reserve(num_elements);

                    // compress the matrix
                    compress(reinterpret_cast<const T*>(data), &memory[0], extent, ld_data, bs, this);

                    // set up the internal pointer to the compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(&memory[0]);
                }
            }

            template <typename TT>
            matrix(const std::vector<TT>& data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                matrix(&data[0], extent, ld_data, bs)
            {
                ;
            }

            matrix(matrix&& rhs) = default;

            virtual ~matrix()
            { 
                compressed_data = nullptr;
            }
   
            static ptrdiff_t compress(const T* data, fp_type* compressed_data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default, matrix* mat = nullptr)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::compress: any of the pointers is a nullptr" << std::endl;
                    return 0;
                }

                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0)
                {
                    return 0;
                }

                const std::size_t num_elements_a = (mat != nullptr ? mat->num_elements_a : fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs));
                const std::size_t num_elements_b = (mat != nullptr ? mat->num_elements_b : fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs)));
                const std::size_t num_elements_c = (mat != nullptr ? mat->num_elements_c : fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * bs));
                const std::size_t num_elements_d = (mat != nullptr ? mat->num_elements_d : fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * (n - (n / bs) * bs)));

                // compress the matrix block by block
                alignas(alignment) T buffer[bs * bs];
                fp_type* ptr = compressed_data;

                for (std::size_t j = 0; j < m; j += bs)
                {
                    for (std::size_t i = 0; i < n; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // copy blocks into the 'buffer'
                        for (std::size_t jj = 0; jj < mm; ++jj)
                        {
                            for (std::size_t ii = 0; ii < nn; ++ii)
                            {
                                buffer[idx<L>(jj, ii, mm, nn)] = data[idx<L>(j + jj, i + ii, ld_data, ld_data)];
                            }  
                        }

                        // compress the 'buffer'
                        fp<T>::template compress<BE, BM>(ptr, &buffer[0], mm * nn);
                        
                        // move on to the next block
                        ptr += ((n - i) < bs ? num_elements_b : ((m - j) < bs ? num_elements_c : num_elements_a));
                    }
                }

                if (num_elements_d != 0)
                {
                    ptr = ptr - (num_elements_b != 0 ? num_elements_b : num_elements_c) + num_elements_d;
                }

                return (ptr - compressed_data);
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0)
                {
                    return 0;
                }

                const std::size_t num_elements_a = fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs);
                const std::size_t num_elements_b = fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs));
                const std::size_t num_elements_c = fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * bs);
                const std::size_t num_elements_d = fp<T>::template format<BE, BM>::memory_footprint_elements((m - (m / bs) * bs) * (n - (n / bs) * bs));

                const std::size_t num_blocks_a = (m / bs) * (n / bs);
                const std::size_t num_blocks_b = (m / bs) * (((n + bs - 1) / bs) - (n / bs));
                const std::size_t num_blocks_c = (((m + bs - 1) / bs) - (m / bs)) * (n / bs);
                const std::size_t num_blocks_d = (((m + bs - 1) / bs) - (m / bs)) * (((n + bs - 1) / bs) - (n / bs));

                return (num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d);
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements(extent, bs) * sizeof(fp_type);
            }
            
            virtual std::size_t memory_footprint_elements() const
            {
                return num_elements;
            }

            virtual std::size_t memory_footprint_bytes() const
            {
                return memory_footprint_elements() * sizeof(fp_type);
            }

            //! \brief General matrix vector multiply
            //!
            //! Computes y = alpha * A(T) * x + beta * y
            virtual void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (m == 0 || n == 0)
                {
                    return;
                } 

                blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                { 
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];

                    #if defined(FP_MKL_INTEGER_GEMM_AVAILABLE)
                    std::vector<MKL_INT8> i8x(transpose ? m : n);
                    alignas(alignment) MKL_INT i32y[bs];

                    T c, d[(transpose ? m : n) / bs + 1];
                    if (BE == 0 && BM == 7)
                    {
                        // 8 bit fixed point representation of the input
                        T x_absmax = f_0;
                        for (std::size_t i = 0; i < (transpose ? m : n); ++i)
                        {
                            x_absmax = std::max(std::abs(x[i]), x_absmax);
                        }
                        c = 0x7F / x_absmax;
                        for (std::size_t i = 0; i < (transpose ? m : n); ++i)
                        {
                            i8x[i] = c * x[i];
                        }

                        for (std::size_t i = 0, k = 0; i < (transpose ? m : n); i += bs, ++k)
                        {
                            d[k] = f_0;
                            const std::size_t ii_max = std::min((transpose ? m : n) - i, bs);
                            for (std::size_t ii = 0; ii < ii_max; ++ii)
                            {
                                d[k] += x[i + ii];
                            }
                        }
                    }
                    #endif

                    // apply matrix to 'x' and add the result to 'y'
                    for (std::size_t j = 0, k = 0; j < m; j += bs)
                    {
                        const std::size_t k_inc = ((m - j) < bs ? num_elements_c : num_elements_a);

                        for (std::size_t i = 0; i < n; i += bs)
                        {
                            const std::size_t mm = std::min(m - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                            #if defined(FP_MKL_INTEGER_GEMM_AVAILABLE)
                            if (BE == 0 && BM == 7)
                            {
                                const float* fptr = reinterpret_cast<const float*>(&compressed_data[k]);
                                const T a = fptr[0];
                                const T b = f_1 / fptr[1];
                                const MKL_INT8* i8a = reinterpret_cast<const MKL_INT8*>(&fptr[2]);

                                // move on to the next block  and prefetch data
                                k += ((n - i) < bs ? num_elements_b : k_inc);
                                #if defined(FP_PREFETCH)
                                if (!(j > (m - bs) && i > (n - bs)))
                                {
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif
                                
                                // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                const std::size_t src_idx = (transpose ? j : i);
                                const std::size_t dst_idx = (transpose ? i : j);
                                const std::size_t M = 1;
                                const std::size_t N = (transpose ? nn : mm); 
                                const std::size_t K = (transpose ? mm : nn);
                                const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                const std::size_t ldx = (L == matrix_layout::rowmajor ? 1 : (transpose ? mm : nn));
                                const std::size_t ldy = (L == matrix_layout::rowmajor ? (transpose ? nn : mm) : 1);
                                const MKL_INT dummy = 0;
                                
                                cblas_gemm_s8u8s32(cblas_layout, CblasTrans, (transpose ? CblasNoTrans : CblasTrans), CblasFixOffset, M, N, K, f_1, &i8x[src_idx], ldx, 0, &i8a[0], lda, 0, f_0, &i32y[0], ldy, &dummy);
                                
                                const T e = f_1 / (b * c);
                                const T f = a * d[src_idx / bs];
                                for (std::size_t jj = 0; jj < (transpose ? nn : mm); ++jj)
                                {
                                    y[dst_idx + jj] += alpha * (i32y[jj] * e + f);
                                }
                            }
                            else                            
                            #endif
                            {
                                // decompress the block
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                                // move on to the next block  and prefetch data
                                k += ((n - i) < bs ? num_elements_b : k_inc);
                                #if defined(FP_PREFETCH)
                                if (!(j > (m - bs) && i > (n - bs)))
                                {
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif

                                // apply general blas matrix vector multiplication
                                const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                const std::size_t src_idx = (transpose ? j : i);
                                const std::size_t dst_idx = (transpose ? i : j);
                                gemv(cblas_layout, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], lda, &x[src_idx], 1, f_1, &y[dst_idx], 1);
                            }
                        }
                    }
                }, transpose, alpha, x, beta, y);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }

            void symmetric_matrix_vector(const T alpha, const T* x, const T beta, T* y) const
            {
                // TODO remove this somehow
            }

            void symmetric_matrix_vector(const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                // TODO remove this somehow
            }
        };

        //! \brief Triangular matrix
        //!
        //! \tparam T data type (of the matrix elements)
        //! \tparam BE number of bits in the exponent
        //! \tparam BM number of bits in the mantissa
        template <typename T, matrix_layout L = matrix_layout::rowmajor, triangular_matrix_type MT = triangular_matrix_type::upper, std::uint32_t BE = fp<T>::default_bits_exponent(), std::uint32_t BM = fp<T>::default_bits_mantissa()>
        class triangular_matrix : public matrix<T, L, BE, BM>
        {
            static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

        public:

            // extent of the matrix
            using matrix<T, L, BE, BM>::n;

            // fp type for data compression
            using fp_type = typename fp<T>::template format<BE, BM>::type;
            using fp_remapped_type = typename FP_NAMESPACE::internal::fp_remap<T, BE, BM>::type;
            
        private:

            // block size
            using matrix<T, L, BE, BM>::bs;

            // matrix format
            static constexpr CBLAS_LAYOUT cblas_layout = (L == matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

            // compresse matrix
            using matrix<T, L, BE, BM>::memory;
            using matrix<T, L, BE, BM>::compressed_data;

            using matrix<T, L, BE, BM>::num_blocks_a;
            using matrix<T, L, BE, BM>::num_elements_a;

            using matrix<T, L, BE, BM>::num_blocks_b;
            using matrix<T, L, BE, BM>::num_elements_b;

            using matrix<T, L, BE, BM>::num_blocks_c;
            using matrix<T, L, BE, BM>::num_elements_c;

            using matrix<T, L, BE, BM>::num_blocks_d;
            using matrix<T, L, BE, BM>::num_elements_d;

            using matrix<T, L, BE, BM>::num_elements;

            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // determine offset from block ID
            std::size_t get_offset(const triangular_matrix_type mt, const std::size_t bj, const std::size_t bi) const
            {
                if (mt == triangular_matrix_type::upper)
                {
                    const std::size_t n_ab_row = (n / bs);
                    const std::size_t n_c_row = ((n + bs - 1) / bs) - n_ab_row;
                    const std::size_t n_abc_row = n_ab_row + n_c_row;
                    const std::size_t n_total = (n_abc_row * (n_abc_row + 1)) / 2;
                    const std::size_t n_abc = n_total - ((n_abc_row - bj) * ((n_abc_row - bj + 1))) / 2;
                    const std::size_t n_a = bj + (bi > bj ? 1 : 0);
                    const std::size_t n_b = n_abc - bj * (1 + n_c_row) + (bi > (bj + 1) ? (bi - (bj + 1)) : 0);
                    const std::size_t n_c = bj * n_c_row;
                    
                    return (n_a * num_elements_a + n_b * num_elements_b + n_c * num_elements_c);
                }
                else
                {
                    const std::size_t n_blocks = (n + bs - 1) / bs;
                    const std::size_t n_a = bj;
                    // the computation of 'n_b' and 'n_c' assumes 'num_elements_c != 0'
                    const std::size_t n_b = (bj * (bj + 1)) / 2 - bj + (bj < (n_blocks - 1) ? bi : 0);
                    const std::size_t n_c = (bj == (n_blocks - 1) ? bi : 0);

                    // fix 'num_elements_c == 0' case
                    return (n_a * num_elements_a + n_b * num_elements_b + n_c * (num_elements_c != 0 ? num_elements_c : num_elements_b));
                }
            }

        public:

            // (default) block size
            static constexpr std::size_t bs_default = matrix<T, L, BE, BM>::bs_default;

            // constructor
            triangular_matrix() = delete;

            template <typename TT>
            triangular_matrix(const TT* data, const std::array<std::size_t, 1>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                matrix<T, L, BE, BM>(data, extent, ld_data, MT, bs)
            {
                if (std::is_same<TT, fp_type>::value)
                {
                    // 'data' points to a compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(data);
                }
                else
                {
                    // allocate memory for the compressed matrix
                    memory.reserve(num_elements);

                    // compress the matrix
                    compress(reinterpret_cast<const T*>(data), &memory[0], extent, ld_data, bs, this);

                    // set up the internal pointer to the compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(&memory[0]);
                }
            }

            template <typename TT>
            triangular_matrix(const std::vector<TT>& data, const std::array<std::size_t, 1>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                triangular_matrix(&data[0], extent, ld_data, bs)
            {
                ;
            }

            template <typename TT>
            triangular_matrix(const TT* data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                triangular_matrix(data, std::array<std::size_t, 1>({extent[0]}), ld_data, bs)
            {
                ;
            }

            template <typename TT>
            triangular_matrix(const std::vector<TT>& data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default)
                :
                triangular_matrix(&data[0], std::array<std::size_t, 1>({extent[0]}), ld_data, bs)
            {
                ;
            }

            triangular_matrix(triangular_matrix&& rhs) = default;

            virtual ~triangular_matrix() 
            { 
                compressed_data = nullptr; 
            }

            static ptrdiff_t compress(const T* data, fp_type* compressed_data, const std::array<std::size_t, 1>& extent, const std::size_t ld_data, const std::size_t bs = bs_default, triangular_matrix* mat = nullptr)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::compress: any of the pointers is a nullptr" << std::endl;
                    return 0;
                }

                const std::size_t n = extent[0];
                if (n == 0)
                {
                    return 0;
                }

                const std::size_t num_elements_a = (mat != nullptr ? mat->num_elements_a : fp<T>::template format<BE, BM>::memory_footprint_elements((bs * (bs + 1)) / 2));
                const std::size_t num_elements_b = (mat != nullptr ? mat->num_elements_b : fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs));
                const std::size_t num_elements_c = (mat != nullptr ? mat->num_elements_c : fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs)));
                const std::size_t num_elements_d = (mat != nullptr ? mat->num_elements_d : fp<T>::template format<BE, BM>::memory_footprint_elements(((n - (n / bs) * bs) * (n - (n / bs) * bs + 1)) / 2));
                
                // compress the matrix block by block
                constexpr bool upper_rowmajor = (MT == triangular_matrix_type::upper) && (L == matrix_layout::rowmajor);
                constexpr bool lower_colmajor = (MT == triangular_matrix_type::lower) && (L == matrix_layout::colmajor);

                alignas(alignment) T buffer[bs * bs];
                fp_type* ptr = compressed_data;
                for (std::size_t j = 0; j < n; j += bs)
                {
                    const std::size_t i_start = (MT == triangular_matrix_type::upper ? j : 0);
                    const std::size_t i_end = (MT == triangular_matrix_type::upper ? n : (j + 1));
                    
                    for (std::size_t i = i_start; i < i_end; i += bs)
                    {
                        const std::size_t mm = std::min(n - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // copy blocks into the 'buffer'
                        for (std::size_t jj = 0, kk = 0; jj < mm; ++jj)
                        {
                            // diagonal blocks
                            if (i == j)
                            {
                                const std::size_t ii_start = (upper_rowmajor || lower_colmajor ? jj : 0);
                                const std::size_t ii_end = (upper_rowmajor || lower_colmajor ? nn : (jj + 1));

                                for (std::size_t ii = ii_start; ii < ii_end; ++ii, ++kk)
                                {
                                    buffer[kk] = data[(j + jj) * ld_data + (i + ii)];
                                }
                            }
                            // non-diagonal blocks
                            else
                            {
                                for (std::size_t ii = 0; ii < nn; ++ii)
                                {
                                    buffer[idx<L>(jj, ii, mm, nn)] = data[idx<L>(j + jj, i + ii, ld_data, ld_data)];
                                }
                            }
                        }    

                        // compress the 'buffer'
                        fp<T>::template compress<BE, BM>(ptr, buffer, (i == j ? ((mm * (mm + 1)) / 2) : mm * nn));

                        // move on to the next block
                        if (i == j)
                        {
                            ptr += num_elements_a;
                        }
                        else
                        {
                            const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                            ptr += ((n - ij) < bs ? num_elements_c : num_elements_b);
                        }
                    }
                }
                
                if (num_elements_d != 0)
                {
                    ptr = ptr - num_elements_a + num_elements_d;
                }

                return (ptr - compressed_data);
            }

            static ptrdiff_t compress(const T* data, fp_type* compressed_data, const std::array<std::size_t, 2>& extent, const std::size_t ld_data, const std::size_t bs = bs_default, triangular_matrix* mat = nullptr)
            {
                return compress(data, compressed_data, std::array<std::size_t, 1>({extent[0]}), ld_data, bs, mat);
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                const std::size_t n = extent[0];
                if (n == 0)
                {
                    return 0;
                }

                const std::size_t num_elements_a = fp<T>::template format<BE, BM>::memory_footprint_elements((bs * (bs + 1)) / 2);
                const std::size_t num_elements_b = fp<T>::template format<BE, BM>::memory_footprint_elements(bs * bs);
                const std::size_t num_elements_c = fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs) * bs));
                const std::size_t num_elements_d = fp<T>::template format<BE, BM>::memory_footprint_elements(((n - (n / bs) * bs) * (n - (n / bs) * bs + 1)) / 2);

                const std::size_t num_blocks_a = n / bs;
                const std::size_t num_blocks_b = (((n / bs) * ((n / bs) + 1)) / 2) - (n / bs);
                const std::size_t num_blocks_c = (n / bs) * (((n + bs - 1) / bs) - (n / bs));
                const std::size_t num_blocks_d = ((n + bs - 1) / bs) - (n / bs);

                return (num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d);
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements(std::array<std::size_t, 1>({extent[0]}), bs);
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements(extent, bs) * sizeof(fp_type);
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_bytes(std::array<std::size_t, 1>({extent[0]}), bs);
            }

            virtual std::size_t memory_footprint_elements() const
            {
                return num_elements;
            }

            virtual std::size_t memory_footprint_bytes() const
            {
                return memory_footprint_elements() * sizeof(fp_type);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (n == 0)
                {
                    return;
                }

                matrix<T, L, BE, BM>::blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                { 
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];
                    alignas(alignment) T buffer_y[bs];
                    
                    // apply matrix to 'x': diagonal blocks first
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == triangular_matrix_type::upper ? j : 0);
                        const std::size_t i_end = (MT == triangular_matrix_type::upper ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t nn = std::min(n - i, bs);
                        
                            // consider diagonal block
                            if (i == j)
                            {
                                // decompress the 'buffer'
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], (nn * (nn + 1)) / 2);    
                                
                                // prepare call to tpmv
                                for (std::size_t jj = 0; jj < nn; ++jj)
                                {
                                    buffer_y[jj] = x[j + jj];
                                }

                                // apply triangular matrix vector multiply
                                tpmv(cblas_layout, (MT == triangular_matrix_type::upper ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, nn, &buffer_a[0], &buffer_y[0], 1);

                                // scale by 'alpha'
                                for (std::size_t jj = 0; jj < nn; ++jj)
                                {
                                    y[j + jj] += alpha * buffer_y[jj];
                                }

                                // move on to the next block
                                k += num_elements_a;
                            }
                            // skip non-diagonal blocks
                            else
                            {
                                const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                                k += ((n - ij) < bs ? num_elements_c : num_elements_b);
                            }
                        }
                    }

                    // apply matrix to 'x': non-diagonal blocks
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == triangular_matrix_type::upper ? j : 0);
                        const std::size_t i_end = (MT == triangular_matrix_type::upper ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t mm = std::min(n - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                            // skip diagonal blocks
                            if (i == j)
                            {
                                // move to the next block and prefetch data
                                k += num_elements_a;
                                #if defined(FP_PREFETCH)
                                if (j < (n - bs))
                                {
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif
                            }
                            else
                            {
                                // decompress the 'buffer'
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                                // move to the next block and prefetch data
                                const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                                k += ((n - ij) < bs ? num_elements_c : num_elements_b);
                                #if defined(FP_PREFETCH)
                                __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                #endif

                                // apply blas matrix vector multiplication
                                const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                const std::size_t src_idx = (transpose ? j : i);
                                const std::size_t dst_idx = (transpose ? i : j);
                                gemv(cblas_layout, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], lda, &x[src_idx], 1, f_1, &y[dst_idx], 1);
                            }
                        }
                    }
                }, transpose, alpha, x, beta, y);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }

            virtual void symmetric_matrix_vector(const T alpha, const T* x, const T beta, T* y) const
            {
                if (n == 0)
                {
                    return;
                }

                matrix<T, L, BE, BM>::blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                {
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];

                    // apply symmetric matrix
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == triangular_matrix_type::upper ? j : 0);
                        const std::size_t i_end = (MT == triangular_matrix_type::upper ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t mm = std::min(n - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                            // diagonal blocks
                            if (i == j)
                            {
                                // decompress the 'buffer'
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], (nn * (nn + 1)) / 2);

                                // move on to the next block and prefetch data
                                k += num_elements_a;
                                #if defined(FP_PREFETCH)
                                if (j < (n - bs))
                                {
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif

                                // apply symmetric matrix vector multiply
                                spmv(cblas_layout, (MT == triangular_matrix_type::upper ? CblasUpper : CblasLower), nn, alpha, &buffer_a[0], &x[i], 1, f_1, &y[i], 1);
                            }
                            // non-diagonal blocks
                            else
                            {
                                // decompress the 'buffer'
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                                // move on to the next block
                                const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                                k += ((n - ij) < bs ? num_elements_c : num_elements_b);
                                #if defined(FP_PREFETCH)
                                __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[k]), 0, FP_PREFETCH_LEVEL);
                                #endif

                                // apply general matrix vector multiplication
                                const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                gemv(cblas_layout, CblasNoTrans, mm, nn, alpha, &buffer_a[0], lda, &x[i], 1, f_1, &y[j], 1);
                                gemv(cblas_layout, CblasTrans, mm, nn, alpha, &buffer_a[0], lda, &x[j], 1, f_1, &y[i], 1);

                            }
                        }
                    }
                }, false, alpha, x, beta, y);
            }

            virtual void symmetric_matrix_vector(const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                symmetric_matrix_vector(alpha, &x[0], beta, &y[0]);
            }

            void solve(const bool transpose, const T alpha, T* x, const T* y) const
            {
                if (n == 0)
                {
                    return;
                }

                matrix<T, L, BE, BM>::blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                {
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];
                    alignas(alignment) T buffer_x[bs];

                    if ((transpose && MT == triangular_matrix_type::upper) ||
                        (!transpose && MT == triangular_matrix_type::lower))
                    {
                        const std::size_t n_blocks = (n + bs - 1) / bs;
                        for (std::size_t bj = 0; bj < n_blocks; ++bj)
                        {
                            const std::size_t mm = std::min(n - bj * bs, bs);
                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                buffer_x[jj] = f_0;
                            }

                            for (std::size_t bi = 0; bi < bj; ++bi)
                            {
                                const std::size_t nn = std::min(n - bi * bs, bs);

                                // decompress the 'buffer'
                                const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                                #if defined(FP_PREFETCH)
                                // prefetch the next block
                                if ((bi + 1) < bj)
                                {
                                    const std::size_t next_k = (transpose ? get_offset(MT, bi + 1, bj) : get_offset(MT, bj, bi + 1));
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[next_k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif

                                // apply general matrix vector multiplication
                                if (transpose)
                                {
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? mm : nn);
                                    gemv(cblas_layout, CblasTrans, nn, mm, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                }
                                else
                                {
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                    gemv(cblas_layout, CblasNoTrans, mm, nn, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                }
                            }

                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                y[bj * bs + jj] = x[bj * bs + jj] - buffer_x[jj];
                            }

                            // decompress the 'buffer'
                            const std::size_t k = get_offset(MT, bj, bj);
                            fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], (mm * (mm + 1)) / 2);

                            // apply triangular solve 
                            tpsv(cblas_layout, (MT == triangular_matrix_type::upper ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, mm, &buffer_a[0], &y[bj * bs], 1);
                        }
                    }
                    else if ((!transpose && MT == triangular_matrix_type::upper) ||
                            (transpose && MT == triangular_matrix_type::lower))
                    {
                        const std::size_t ij_start = (n + bs - 1) / bs - 1;
                        for (std::size_t bj = ij_start; bj >= 0; --bj)
                        {
                            const std::size_t mm = std::min(n - bj * bs, bs);
                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                buffer_x[jj] = f_0;
                            }

                            const std::size_t i_stop = bj + 1;
                            for (std::size_t bi = ij_start; bi >= i_stop; --bi)
                            {
                                const std::size_t nn = std::min(n - bi * bs, bs);

                                // decompress the 'buffer'
                                const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                                #if defined(FP_PREFETCH)
                                // prefetch the next block
                                if (bi > i_stop)
                                {
                                    const std::size_t next_k = (transpose ? get_offset(MT, bi - 1, bj) : get_offset(MT, bj, bi - 1));
                                    __builtin_prefetch(reinterpret_cast<const void*>(&compressed_data[next_k]), 0, FP_PREFETCH_LEVEL);
                                }
                                #endif

                                // apply general matrix vector multiplication
                                if (transpose)
                                {
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? mm : nn);
                                    gemv(cblas_layout, CblasTrans, nn, mm, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                }
                                else
                                {
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                    gemv(cblas_layout, CblasNoTrans, mm, nn, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                }

                                if (bi == i_stop)
                                {
                                    break;
                                }
                            }

                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                y[bj * bs + jj] = x[bj * bs + jj] - buffer_x[jj];
                            }

                            // decompress the 'buffer'
                            const std::size_t k = get_offset(MT, bj, bj);
                            fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], (mm * (mm + 1)) / 2);

                            // apply triangular solve 
                            tpsv(cblas_layout, (MT == triangular_matrix_type::upper ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, mm, &buffer_a[0], &y[bj * bs], 1);

                            if (bj == 0)
                            {
                                break;
                            }
                        }
                    }

                    // scale with 1 / alpha
                    const T inv_alpha = f_1 / alpha;
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        y[j] *= inv_alpha;
                    }
                }, transpose, alpha, y, f_0, x);
            }

            void solve(const bool transpose, const T alpha, std::vector<T>& x, const std::vector<T>& y) const
            {
                solve(transpose, alpha, &x[0], &y[0]);
            }
        };
    }
}

#endif
