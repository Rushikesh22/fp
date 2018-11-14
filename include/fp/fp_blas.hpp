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
#include <omp.h>

#if !defined(FP_NAMESPACE)
    #define FP_NAMESPACE fw
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
        static std::size_t idx(const std::size_t j, const std::size_t i, const std::size_t m, const std::size_t n);

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
    }
}

#include <fp/fp.hpp>
#include <blas/wrapper.hpp>

namespace FP_NAMESPACE
{
	namespace blas
	{
        //! matrix types: general, lower triangular, upper triangular
        enum class matrix_type { general = 0, triangular = 1, lower_triangular = 2, upper_triangular = 3 };

        //! \brief General matrix
        //!
        //! \tparam T data type (of the matrix elements)
        //! \tparam BE number of bits in the exponent
        //! \tparam BM number of bits in the mantissa
        template <typename T, matrix_layout L = matrix_layout::rowmajor, std::uint32_t BM = ieee754_fp<T>::bm, std::uint32_t BE = ieee754_fp<T>::be>
        class matrix_base
        {
            static_assert(std::is_floating_point<T>::value, "error: only floating point numbers are allowed");

        public:

            // extent of the matrix: 'm' rows and 'n' columns
            const std::size_t m;
            const std::size_t n;

            // fp type for data compression
            using fp_type = typename fp_stream<BM, BE>::type;

            // (default) block size
            static constexpr std::size_t bs_default = 32;

        protected:

            // block size
            const std::size_t bs;

            // compressed matrix
            std::vector<fp_type> memory;
            const fp_type* compressed_data;

            // partitioning
            struct partition_t
            {
                const std::size_t num_elements_a;
                const std::size_t num_elements_b;
                const std::size_t num_elements_c;
                const std::size_t num_elements_d;
                const std::size_t num_elements;
            } const partition;

            template <matrix_type MT>
            static partition_t make_partition(const std::array<std::size_t, 2>& extent, const std::size_t bs)
            {
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0)
                {
                    return {{}};
                }

                if (MT == matrix_type::general)
                {
                    const std::size_t num_elements_a = fp_stream<BM, BE>::memory_footprint_elements(bs * bs);
                    const std::size_t num_elements_b = fp_stream<BM, BE>::memory_footprint_elements(bs * (n - (n / bs) * bs));
                    const std::size_t num_elements_c = fp_stream<BM, BE>::memory_footprint_elements((m - (m / bs) * bs) * bs);
                    const std::size_t num_elements_d = fp_stream<BM, BE>::memory_footprint_elements((m - (m / bs) * bs) * (n - (n / bs) * bs));
                    const std::size_t num_blocks_a = (m / bs) * (n / bs);
                    const std::size_t num_blocks_b = (m / bs) * (((n + bs - 1) / bs) - (n / bs));
                    const std::size_t num_blocks_c = (((m + bs - 1) / bs) - (m / bs)) * (n / bs);
                    const std::size_t num_blocks_d = (((m + bs - 1) / bs) - (m / bs)) * (((n + bs - 1) / bs) - (n / bs));
                    return { num_elements_a, num_elements_b, num_elements_c, num_elements_d,
                        num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d };
                }
                else
                {
                    const std::size_t num_elements_a = fp_stream<BM, BE>::memory_footprint_elements((bs * (bs + 1)) / 2);
                    const std::size_t num_elements_b = fp_stream<BM, BE>::memory_footprint_elements(bs * bs);
                    const std::size_t num_elements_c = fp_stream<BM, BE>::memory_footprint_elements(bs * (n - (n / bs) * bs));
                    const std::size_t num_elements_d = fp_stream<BM, BE>::memory_footprint_elements(((n - (n / bs) * bs) * (n - (n / bs) * bs + 1)) / 2);
                    const std::size_t num_blocks_a = (n / bs);
                    const std::size_t num_blocks_b = (((n / bs) * ((n / bs) + 1)) / 2) - (n / bs);
                    const std::size_t num_blocks_c = (n / bs) * (((n + bs - 1) / bs) - (n / bs));
                    const std::size_t num_blocks_d = ((n + bs - 1) / bs) - (n / bs);
                    return { num_elements_a, num_elements_b, num_elements_c, num_elements_d,
                        num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d };
                }
            }

            // constructor for triangular matrices
            template <typename TT>
            matrix_base(const TT* data, const std::size_t ld_data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
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
                partition(make_partition<matrix_type::triangular>({m, n}, bs))
            {
                ;
            }

            // constructor for general matrices
            template <typename TT>
            matrix_base(const TT* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                m(extent[0]),
                n(extent[1]),
                bs(bs),
                compressed_data(nullptr),
                //  a a a | b
                //  a a a | b
                // -------+---
                //  c c c | d
                partition(make_partition<matrix_type::general>({m, n}, bs))
            {
                ;
            }

            // compression
            template <matrix_type MT>
            static ptrdiff_t compress(const T* data, const std::size_t ld_data, fp_type* compressed_data, const std::array<std::size_t, 2>& extent, const std::size_t bs, const partition_t& partition)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in matrix_base<..," << BE << "," << BM << ">::compress: any of the pointers is a nullptr" << std::endl;
                    return 0;
                }

                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0)
                {
                    return 0;
                }

                constexpr bool upper_rowmajor = (MT == matrix_type::upper_triangular) && (L == matrix_layout::rowmajor);
                constexpr bool lower_colmajor = (MT == matrix_type::lower_triangular) && (L == matrix_layout::colmajor);

                // compress the matrix block by block
                alignas(alignment) T buffer[bs * bs];
                fp_type* ptr = compressed_data;

                for (std::size_t j = 0; j < m; j += bs)
                {
                    const std::size_t i_start_triangular = (MT == matrix_type::upper_triangular ? j : 0);
                    const std::size_t i_end_triangular = (MT == matrix_type::upper_triangular ? n : (j + 1));

                    const std::size_t i_start = (MT == matrix_type::general ? 0 : i_start_triangular);
                    const std::size_t i_end = (MT == matrix_type::general ? n : i_end_triangular);
                    
                    for (std::size_t i = i_start; i < i_end; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // copy blocks into the 'buffer'
                        if (MT != matrix_type::general && i == j)
                        {
                            // diagonal blocks
                            for (std::size_t jj = 0, kk = 0; jj < mm; ++jj)
                            {
                                const std::size_t ii_start = (upper_rowmajor || lower_colmajor ? jj : 0);
                                const std::size_t ii_end = (upper_rowmajor || lower_colmajor ? nn : (jj + 1));

                                for (std::size_t ii = ii_start; ii < ii_end; ++ii, ++kk)
                                {
                                    buffer[kk] = data[(j + jj) * ld_data + (i + ii)];
                                }
                            }
                        }
                        else
                        {
                            // non-diagonal blocks
                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                for (std::size_t ii = 0; ii < nn; ++ii)
                                {
                                    buffer[idx<L>(jj, ii, mm, nn)] = data[idx<L>(j + jj, i + ii, ld_data, ld_data)];
                                }
                            }
                        }

                        // compress the 'buffer'
                        if (MT == matrix_type::general)
                        {
                            fp_stream<BM, BE>::compress(buffer, ptr, mm * nn);
                            // move on to the next block
                            ptr += ((n - i) < bs ? partition.num_elements_b : ((m - j) < bs ? partition.num_elements_c : partition.num_elements_a));
                        }
                        else
                        {
                            fp_stream<BM, BE>::compress(buffer, ptr, (i == j ? ((mm * (mm + 1)) / 2) : mm * nn));
                            // move on to the next block
                            if (i == j)
                            {
                                ptr += partition.num_elements_a;
                            }
                            else
                            {   
                                const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                ptr += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                            }
                        }
                    }
                }
                
                if (partition.num_elements_d != 0)
                {
                    if (MT == matrix_type::general)
                    {
                        ptr = ptr - (partition.num_elements_b != 0 ? partition.num_elements_b : partition.num_elements_c) + partition.num_elements_d;
                    }
                    else
                    {
                        ptr = ptr - partition.num_elements_a + partition.num_elements_d;
                    }
                }

                return (ptr - compressed_data);
            }

            // decompression
            template <matrix_type MT>
            static void decompress(const fp_type* compressed_data, T* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs, const partition_t& partition)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in matrix_base<..," << BE << "," << BM << ">::decompress: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0)
                {
                    return;
                }

                // decompress the matrix block by block
                constexpr bool upper_rowmajor = (MT == matrix_type::upper_triangular) && (L == matrix_layout::rowmajor);
                constexpr bool lower_colmajor = (MT == matrix_type::lower_triangular) && (L == matrix_layout::colmajor);

                alignas(alignment) T buffer[bs * bs];
                const fp_type* ptr = compressed_data;
                for (std::size_t j = 0; j < n; j += bs)
                {
                    const std::size_t i_start_triangular = (MT == matrix_type::upper_triangular ? j : 0);
                    const std::size_t i_end_triangular = (MT == matrix_type::upper_triangular ? n : (j + 1));

                    const std::size_t i_start = (MT == matrix_type::general ? 0 : i_start_triangular);
                    const std::size_t i_end = (MT == matrix_type::general ? n : i_end_triangular);
                    
                    for (std::size_t i = i_start; i < i_end; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // decompress the 'buffer'
                        if (MT == matrix_type::general)
                        {
                            fp_stream<BM, BE>::decompress(ptr, &buffer[0], mm * nn);
                            // move on to the next block
                            ptr += ((n - i) < bs ? partition.num_elements_b : ((m - j) < bs ? partition.num_elements_c : partition.num_elements_a));
                        }
                        else
                        {
                            fp_stream<BM, BE>::decompress(ptr, buffer, (i == j ? ((mm * (mm + 1)) / 2) : mm * nn));
                            // move on to the next block
                            if (i == j)
                            {
                                ptr += partition.num_elements_a;
                            }
                            else
                            {
                                const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                ptr += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                            }
                        }

                        // output the 'buffer'
                        if (MT != matrix_type::general && i == j)
                        {
                            // diagonal blocks
                            for (std::size_t jj = 0, kk = 0; jj < mm; ++jj)
                            {
                                const std::size_t ii_start = (upper_rowmajor || lower_colmajor ? jj : 0);
                                const std::size_t ii_end = (upper_rowmajor || lower_colmajor ? nn : (jj + 1));

                                for (std::size_t ii = ii_start; ii < ii_end; ++ii, ++kk)
                                {
                                    data[(j + jj) * ld_data + (i + ii)] = buffer[kk];
                                }
                            }    
                        }
                        else
                        {
                            // non-diagonal blocks
                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                for (std::size_t ii = 0; ii < nn; ++ii)
                                {
                                    data[idx<L>(j + jj, i + ii, ld_data, ld_data)] = buffer[idx<L>(jj, ii, mm, nn)];
                                }
                            }
                        }
                    }
                }
            }

            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // template function providing a frame for different blas2 kernel implementations
            template <typename F>
            void blas2_frame(const F& kernel, const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (n == 0 || m == 0)
                {
                    return;
                }

                const std::size_t mn = (transpose ? n : m);

                // handle some special cases
                if (alpha == f_0)
                {
                    if (beta == f_0)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < mn; ++j)
                        {
                            y[j] = f_0;
                        }
                    }
                    else if (beta != f_1)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < mn; ++j)
                        {
                            y[j] = beta * y[j];
                        }
                    }

                    return;
                }

                // allocate local memory
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

            matrix_base() = delete;

        public:

            virtual std::size_t memory_footprint_elements() const
            {
                return partition.num_elements;
            }

            virtual std::size_t memory_footprint_bytes() const
            {
                return memory_footprint_elements() * sizeof(fp_type);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const = 0;
        };

        //! \brief General matrix
        //!
        //! \tparam T data type (of the matrix elements)
        //! \tparam BE number of bits in the exponent
        //! \tparam BM number of bits in the mantissa
        template <typename T, matrix_layout L = matrix_layout::rowmajor, std::uint32_t BM = ieee754_fp<T>::bm, std::uint32_t BE = ieee754_fp<T>::be>
        class matrix : public matrix_base<T, L, BM, BE>
        {
            static_assert(std::is_floating_point<T>::value, "error: only floating point numbers are allowed");

            using this_class = matrix<T, L, BM, BE>;
            using base_class = matrix_base<T, L, BM, BE>;

            static constexpr CBLAS_LAYOUT cblas_layout = (L == matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

        public:

            // extent of the matrix: 'm' rows and 'n' columns
            using base_class::m;
            using base_class::n;

            // fp type for data compression
            using fp_type = typename base_class::fp_type;

            // (default) block size
            static constexpr std::size_t bs_default = base_class::bs_default;

        private:

            // block size
            using base_class::bs;

            // compressed matrix
            using base_class::memory;
            using base_class::compressed_data;

            using partition_t = typename base_class::partition_t;
            using base_class::partition;

            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // methods
            using base_class::blas2_frame;

        public:

            // constructor
            matrix() = delete;

            template <typename TT>
            matrix(const TT* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                base_class(data, ld_data, extent, bs)
            {
                if (ld_data > 0)
                {
                    // allocate memory for the compressed matrix
                    memory.reserve(partition.num_elements);
                    
                    // compress the matrix
                    base_class::template compress<matrix_type::general>(reinterpret_cast<const T*>(data), ld_data, &memory[0], extent, bs, partition);

                    // set up the internal pointer to the compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(&memory[0]);
                }
            }

            template <typename TT>
            matrix(const std::vector<TT>& data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                matrix(&data[0], ld_data, extent, bs)
            {
                ;
            }

            template <typename TT>
            matrix(const TT* data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                matrix(data, 0, extent, bs)
            {
                if (std::is_same<TT, fp_type>::value)
                {
                    // 'data' points to a compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(data);
                }
                else
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::matrix: external pointer type does not match fp_type" << std::endl;
                }
            }

            matrix(matrix&& rhs) = default;

            virtual ~matrix()
            { 
                compressed_data = nullptr;
            }
   
            static ptrdiff_t compress(const T* data, const std::size_t ld_data, fp_type* compressed_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::compress: any of the pointers is a nullptr" << std::endl;
                    return 0;
                }
    
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0) 
                {
                    return 0;
                }

                const partition_t partition = base_class::template make_partition<matrix_type::general>(extent, bs);
                return base_class::template compress<matrix_type::general>(data, ld_data, compressed_data, extent, bs, partition);
            }

            static void decompress(const fp_type* compressed_data, T* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::decompress: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0) 
                {
                    return;
                }
            
                const partition_t partition = base_class::template make_partition<matrix_type::general>(extent, bs);
                base_class::template decompress<matrix_type::general>(compressed_data, data, ld_data, extent, bs, partition);
            }

            void decompress(T* data, const std::size_t ld_data = 0)
            {
                if (data == nullptr)
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::decompress: pointers is a nullptr" << std::endl;
                    return;
                }

                base_class::template decompress<matrix_type::general>(compressed_data, data, (ld_data == 0 ? (L == matrix_layout::rowmajor ? n : m) : ld_data), {m, n}, bs, partition);
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0)
                {
                    return 0;
                }

                return (base_class::template make_partition<matrix_type::general>(extent, bs)).num_elements;
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements(extent, bs) * sizeof(fp_type);
            }

            using base_class::memory_footprint_elements;
            using base_class::memory_footprint_bytes;

            //! \brief General matrix vector multiply
            //!
            //! Computes y = alpha * A(T) * x + beta * y
            virtual void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (x == nullptr || y == nullptr)
                {
                    std::cerr << "error in matrix<..," << BE << "," << BM << ">::matrix_vector: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                if (m == 0 || n == 0)
                {
                    return;
                } 

                blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                { 
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];

                #if defined(FP_INTEGER_GEMV)
                    alignas(alignment) T tmp_y[bs];
                    std::vector<T> rescale_p_2(0);
                    if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                    {
                        rescale_p_2.reserve((transpose ? m : n) / bs + 1);
                        for (std::size_t i = 0, k = 0; i < (transpose ? m : n); i += bs, ++k)
                        {
                            rescale_p_2[k] = f_0;
                            const std::size_t ii_max = std::min((transpose ? m : n) - i, bs);
                            for (std::size_t ii = 0; ii < ii_max; ++ii)
                            {
                                rescale_p_2[k] += x[i + ii];
                            }
                        }
                    }
                #endif

                    // apply matrix to 'x' and add the result to 'y'
                    for (std::size_t j = 0, k = 0; j < m; j += bs)
                    {
                        const std::size_t k_inc = ((m - j) < bs ? partition.num_elements_c : partition.num_elements_a);

                        for (std::size_t i = 0; i < n; i += bs)
                        {
                            const std::size_t mm = std::min(m - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                        #if defined(FP_INTEGER_GEMV)
                            if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                            {
                                const T* fptr = reinterpret_cast<const T*>(&compressed_data[k]);
                                const T rescale_p_3 = fptr[0];
                                const T rescale_p_4 = fptr[1];
                                const fp_type* tmp_a = reinterpret_cast<const fp_type*>(&fptr[2]);

                                // move on to the next block
                                k += ((n - i) < bs ? partition.num_elements_b : k_inc);
                                
                                // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                const std::size_t src_idx = (transpose ? j : i);
                                const std::size_t dst_idx = (transpose ? i : j);
                                blas::gemv(L, transpose, mm, nn, &tmp_a[0], &x[src_idx], &tmp_y[0]);
                                // ..finalize gemm call: rescaling
                                const T a = rescale_p_4;
                                const T b = rescale_p_2[src_idx / bs] * rescale_p_3;
                                for (std::size_t jj = 0; jj < (transpose ? nn : mm); ++jj)
                                {
                                    y[dst_idx + jj] += alpha * (tmp_y[jj] * a + b);
                                }
                            }
                            else                            
                        #endif
                            {
                                // decompress the block
                                fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], mm * nn);

                                // move on to the next block
                                k += ((n - i) < bs ? partition.num_elements_b : k_inc);

                                // apply general blas matrix vector multiplication
                                const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                const std::size_t src_idx = (transpose ? j : i);
                                const std::size_t dst_idx = (transpose ? i : j);
                                blas::gemv(cblas_layout, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], lda, &x[src_idx], 1, f_1, &y[dst_idx], 1);
                            }
                        }
                    }
                }, transpose, alpha, x, beta, y);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }

            void gemv(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                matrix_vector(transpose, alpha, x, beta, y);
            }
            
            void gemv(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }
        };

        //! \brief Triangular matrix
        //!
        //! \tparam T data type (of the matrix elements)
        //! \tparam BE number of bits in the exponent
        //! \tparam BM number of bits in the mantissa
        template <typename T, matrix_layout L = matrix_layout::rowmajor, matrix_type MT = matrix_type::upper_triangular, std::uint32_t BM = ieee754_fp<T>::bm, std::uint32_t BE = ieee754_fp<T>::be>
        class triangular_matrix : public matrix_base<T, L, BM, BE>
        {
            static_assert(std::is_floating_point<T>::value, "error: only floating point numbers are allowed");
            static_assert(MT == matrix_type::upper_triangular || MT == matrix_type::lower_triangular, "error: this should be a triangular matrix");

            using this_class = triangular_matrix<T, L, MT, BM, BE>;
            using base_class = matrix_base<T, L, BM, BE>;

            static constexpr CBLAS_LAYOUT cblas_layout = (L == matrix_layout::rowmajor ? CblasRowMajor : CblasColMajor);

        public:

            // extent of the matrix
            using base_class::n;

            // fp type for data compression
            using fp_type = typename base_class::fp_type;

            // (default) block size
            static constexpr std::size_t bs_default = base_class::bs_default;
            
        private:

            // block size
            using base_class::bs;

            // compresse matrix
            using base_class::memory;
            using base_class::compressed_data;

            using partition_t = typename base_class::partition_t;
            using base_class::partition;
            
            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // determine offset from block ID
            std::size_t get_offset(const matrix_type mt, const std::size_t bj, const std::size_t bi) const
            {
                if (mt == matrix_type::upper_triangular)
                {
                    const std::size_t n_ab_row = (n / bs);
                    const std::size_t n_c_row = ((n + bs - 1) / bs) - n_ab_row;
                    const std::size_t n_abc_row = n_ab_row + n_c_row;
                    const std::size_t n_total = (n_abc_row * (n_abc_row + 1)) / 2;
                    const std::size_t n_abc = n_total - ((n_abc_row - bj) * ((n_abc_row - bj + 1))) / 2;
                    const std::size_t n_a = bj + (bi > bj ? 1 : 0);
                    const std::size_t n_b = n_abc - bj * (1 + n_c_row) + (bi > (bj + 1) ? (bi - (bj + 1)) : 0);
                    const std::size_t n_c = bj * n_c_row;
                    
                    return (n_a * partition.num_elements_a + n_b * partition.num_elements_b + n_c * partition.num_elements_c);
                }
                else
                {
                    const std::size_t n_blocks = (n + bs - 1) / bs;
                    const std::size_t n_a = bj;
                    // the computation of 'n_b' and 'n_c' assumes 'num_elements_c != 0'
                    const std::size_t n_b = (bj * (bj + 1)) / 2 - bj + (bj < (n_blocks - 1) ? bi : 0);
                    const std::size_t n_c = (bj == (n_blocks - 1) ? bi : 0);

                    // fix 'num_elements_c == 0' case
                    return (n_a * partition.num_elements_a + n_b * partition.num_elements_b + n_c * (partition.num_elements_c != 0 ? partition.num_elements_c : partition.num_elements_b));
                }
            }

            // methods
            using base_class::blas2_frame;

        public:

            // constructor
            triangular_matrix() = delete;

            template <typename TT>
            triangular_matrix(const TT* data, const std::size_t ld_data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
                :
                base_class(data, ld_data, extent, bs)
            {
                if (ld_data > 0)
                {
                    // allocate memory for the compressed matrix
                    memory.reserve(partition.num_elements);
                    
                    // compress the matrix
                    base_class::template compress<MT>(reinterpret_cast<const T*>(data), ld_data, &memory[0], {n, n}, bs, partition);

                    // set up the internal pointer to the compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(&memory[0]);
                }
            }

            template <typename TT>
            triangular_matrix(const TT* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                triangular_matrix(data, ld_data, std::array<std::size_t, 1>({extent[0]}), bs)
            {
                ;
            }

            template <typename TT>
            triangular_matrix(const std::vector<TT>& data, const std::size_t ld_data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
                :
                triangular_matrix(&data[0], ld_data, extent, bs)
            {
                ;
            }

            template <typename TT>
            triangular_matrix(const std::vector<TT>& data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                triangular_matrix(&data[0], ld_data, extent, bs)
            {
                ;
            }

            template <typename TT>
            triangular_matrix(const TT* data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
                :
                triangular_matrix(data, 0, extent, bs)
            {
                if (std::is_same<TT, fp_type>::value)
                {
                    // 'data' points to a compressed matrix
                    compressed_data = reinterpret_cast<const fp_type*>(data);
                }
                else
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::triangular_matrix: external pointer type does not match fp_type" << std::endl;
                }
            }

            template <typename TT>
            triangular_matrix(const TT* data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
                :
                triangular_matrix(data, {extent[0]}, bs)
            {
                ;
            }

            triangular_matrix(triangular_matrix&& rhs) = default;

            virtual ~triangular_matrix() 
            { 
                compressed_data = nullptr; 
            }

            static ptrdiff_t compress(const T* data, const std::size_t ld_data, fp_type* compressed_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::compress: any of the pointers is a nullptr" << std::endl;
                    return 0;
                }
                
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0) 
                {
                    return 0;
                }

                const partition_t partition  = base_class::template make_partition<MT>(extent, bs);
                return base_class::template compress<MT>(data, ld_data, compressed_data, extent, bs, partition);
            }

            static void decompress(const fp_type* compressed_data, T* data, const std::size_t ld_data, const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                if (data == nullptr || compressed_data == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::decompress: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0) 
                {
                    return;
                }

                const partition_t partition = base_class::template make_partition<MT>(extent, bs);
                base_class::template decompress<MT>(compressed_data, data, ld_data, extent, bs, partition);
            }

            static ptrdiff_t compress(const T* data, const std::size_t ld_data, fp_type* compressed_data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                return compress(data, ld_data, compressed_data, {extent[0], extent[0]}, bs);
            }

            static void decompress(const fp_type* compressed_data, T* data, const std::size_t ld_data, const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                decompress(compressed_data, data, ld_data, {extent[0], extent[0]}, bs);
            }

            void decompress(T* data, const std::size_t ld_data = 0)
            {
                base_class::template decompress<MT>(compressed_data, data, (ld_data == 0 ? n : ld_data), {n, n}, bs, partition);
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                const std::size_t m = extent[0];
                const std::size_t n = extent[1];
                if (m == 0 || n == 0 || bs == 0)
                {
                    return 0;
                }

                return (base_class::template make_partition<MT>(extent, bs)).num_elements;
            }

            static std::size_t memory_footprint_elements(const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements({extent[0], extent[0]}, bs);
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 2>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_elements(extent, bs) * sizeof(fp_type);
            }

            static std::size_t memory_footprint_bytes(const std::array<std::size_t, 1>& extent, const std::size_t bs = bs_default)
            {
                return memory_footprint_bytes({extent[0], extent[0]}, bs);
            }

            using base_class::memory_footprint_elements;
            using base_class::memory_footprint_bytes;

            virtual void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                if (x == nullptr || y == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::matrix_vector: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                if (n == 0)
                {
                    return;
                }

                blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                { 
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];
                    alignas(alignment) T buffer_y[bs];
                    
                #if defined(FP_INTEGER_GEMV)
                    alignas(alignment) T tmp_y[bs];
                    std::vector<T> rescale_p_2(0);
                    if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                    {
                        rescale_p_2.reserve(n / bs + 1);
                        for (std::size_t i = 0, k = 0; i < n; i += bs, ++k)
                        {
                            rescale_p_2[k] = f_0;
                            const std::size_t ii_max = std::min(n - i, bs);
                            for (std::size_t ii = 0; ii < ii_max; ++ii)
                            {
                                rescale_p_2[k] += x[i + ii];
                            }
                        }
                    }
                #endif
                    
                    // apply matrix to 'x': diagonal blocks first
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == matrix_type::upper_triangular ? j : 0);
                        const std::size_t i_end = (MT == matrix_type::upper_triangular ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t nn = std::min(n - i, bs);
                        
                            // consider diagonal block
                            if (i == j)
                            {
                                // decompress the 'buffer'
                                fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], (nn * (nn + 1)) / 2);    
                                
                                // prepare call to tpmv
                                for (std::size_t jj = 0; jj < nn; ++jj)
                                {
                                    buffer_y[jj] = x[j + jj];
                                }

                                // apply triangular matrix vector multiply
                                blas::tpmv(cblas_layout, (MT == matrix_type::upper_triangular ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, nn, &buffer_a[0], &buffer_y[0], 1);

                                // scale by 'alpha'
                                for (std::size_t jj = 0; jj < nn; ++jj)
                                {
                                    y[j + jj] += alpha * buffer_y[jj];
                                }

                                // move on to the next block
                                k += partition.num_elements_a;
                            }
                            // skip non-diagonal blocks
                            else
                            {
                                const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                k += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                            }
                        }
                    }

                    // apply matrix to 'x': non-diagonal blocks
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == matrix_type::upper_triangular ? j : 0);
                        const std::size_t i_end = (MT == matrix_type::upper_triangular ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t mm = std::min(n - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                            // skip diagonal blocks
                            if (i == j)
                            {
                                // move to the next block
                                k += partition.num_elements_a;
                            }
                            else
                            {
                            #if defined(FP_INTEGER_GEMV)
                                if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                                {
                                    const T* fptr = reinterpret_cast<const T*>(&compressed_data[k]);
                                    const T rescale_p_3 = fptr[0];
                                    const T rescale_p_4 = fptr[1];
                                    const fp_type* tmp_a = reinterpret_cast<const fp_type*>(&fptr[2]);

                                    // move to the next block
                                    const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                    k += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                                    
                                    // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                    const std::size_t src_idx = (transpose ? j : i);
                                    const std::size_t dst_idx = (transpose ? i : j);
                                    // the following gemm call uses alpha=1 internally
                                    blas::gemv(L, transpose, mm, nn, &tmp_a[0], &x[src_idx], &tmp_y[0]);
                                    // ..finalize gemm call: rescaling
                                    const T a = rescale_p_4;
                                    const T b = rescale_p_2[src_idx / bs] * rescale_p_3;
                                    for (std::size_t jj = 0; jj < (transpose ? nn : mm); ++jj)
                                    {
                                        y[dst_idx + jj] += alpha * (tmp_y[jj] * a + b);
                                    }
                                }
                                else                            
                            #endif
                                {
                                    // decompress the 'buffer'
                                    fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], mm * nn);

                                    // move to the next block
                                    const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                    k += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);

                                    // apply blas matrix vector multiplication
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                    const std::size_t src_idx = (transpose ? j : i);
                                    const std::size_t dst_idx = (transpose ? i : j);
                                    blas::gemv(cblas_layout, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], lda, &x[src_idx], 1, f_1, &y[dst_idx], 1);
                                }
                            }
                        }
                    }
                }, transpose, alpha, x, beta, y);
            }

            virtual void matrix_vector(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }

            void tpmv(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
                matrix_vector(transpose, alpha, x, beta, y);
            }

            void tpmv(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }

            void symmetric_matrix_vector(const T alpha, const T* x, const T beta, T* y) const
            {
                if (x == nullptr || y == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::symmetric_matrix_vector: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                if (n == 0)
                {
                    return;
                }

                blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                {
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];

                #if defined(FP_INTEGER_GEMV)
                    alignas(alignment) T tmp_y[bs];
                    std::vector<T> rescale_p_2(0);
                    if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                    {
                        rescale_p_2.reserve(n / bs + 1);

                        for (std::size_t i = 0, k = 0; i < n; i += bs, ++k)
                        {
                            rescale_p_2[k] = f_0;
                            const std::size_t ii_max = std::min(n - i, bs);
                            for (std::size_t ii = 0; ii < ii_max; ++ii)
                            {
                                rescale_p_2[k] += x[i + ii];
                            }
                        }
                    }
                #endif

                    // apply symmetric matrix
                    for (std::size_t j = 0, k = 0; j < n; j += bs)
                    {
                        const std::size_t i_start = (MT == matrix_type::upper_triangular ? j : 0);
                        const std::size_t i_end = (MT == matrix_type::upper_triangular ? n : (j + 1));

                        for (std::size_t i = i_start; i < i_end; i += bs)
                        {
                            const std::size_t mm = std::min(n - j, bs);
                            const std::size_t nn = std::min(n - i, bs);

                            // diagonal blocks
                            if (i == j)
                            {
                                // decompress the 'buffer'
                                fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], (nn * (nn + 1)) / 2);

                                // move on to the next block
                                k += partition.num_elements_a;
                                
                                // apply symmetric matrix vector multiply
                                blas::spmv(cblas_layout, (MT == matrix_type::upper_triangular ? CblasUpper : CblasLower), nn, alpha, &buffer_a[0], &x[i], 1, f_1, &y[i], 1);
                            }
                            // non-diagonal blocks
                            else
                            {
                            #if defined(FP_INTEGER_GEMV)
                                if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                                {
                                    const T* fptr = reinterpret_cast<const T*>(&compressed_data[k]);
                                    const T rescale_p_3 = fptr[0];
                                    const T rescale_p_4 = fptr[1];
                                    const fp_type* tmp_a = reinterpret_cast<const fp_type*>(&fptr[2]);

                                    // move to the next block
                                    const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                    k += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                                    
                                    // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                    //
                                    // the following gemm call uses alpha=1 internally
                                    blas::gemv(L, false, mm, nn, &tmp_a[0], &x[i], &tmp_y[0]);
                                    {
                                        // ..finalize gemm call: rescaling
                                        const T a = rescale_p_4;
                                        const T b = rescale_p_2[i / bs] * rescale_p_3;
                                        for (std::size_t jj = 0; jj < mm; ++jj)
                                        {
                                            y[j + jj] += alpha * (tmp_y[jj] * a + b);
                                        }
                                    }
                                    // the following gemm call uses alpha=1 internally
                                    blas::gemv(L, true, mm, nn, &tmp_a[0], &x[j], &tmp_y[0]);
                                    {
                                        // ..finalize gemm call: rescaling
                                        const T a = rescale_p_4;
                                        const T b = rescale_p_2[j / bs] * rescale_p_3;
                                        for (std::size_t ii = 0; ii < nn; ++ii)
                                        {
                                            y[i + ii] += alpha * (tmp_y[ii] * a + b);
                                        }
                                    }
                                }
                                else                            
                            #endif
                                {
                                    // decompress the 'buffer'
                                    fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], mm * nn);

                                    // move on to the next block
                                    const std::size_t ij = (MT == matrix_type::upper_triangular ? i : j);
                                    k += ((n - ij) < bs ? partition.num_elements_c : partition.num_elements_b);
                                    
                                    // apply general matrix vector multiplication
                                    const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                    blas::gemv(cblas_layout, CblasNoTrans, mm, nn, alpha, &buffer_a[0], lda, &x[i], 1, f_1, &y[j], 1);
                                    blas::gemv(cblas_layout, CblasTrans, mm, nn, alpha, &buffer_a[0], lda, &x[j], 1, f_1, &y[i], 1);
                                }
                            }
                        }
                    }
                }, false, alpha, x, beta, y);
            }

            void symmetric_matrix_vector(const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                symmetric_matrix_vector(alpha, &x[0], beta, &y[0]);
            }

            void spmv(const T alpha, const T* x, const T beta, T* y) const
            {
                symmetric_matrix_vector(alpha, x, beta, y);
            }

            void spmv(const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                symmetric_matrix_vector(alpha, &x[0], beta, &y[0]);
            }

            void triangular_solve(const bool transpose, const T alpha, T* x, const T* y) const
            {
                if (x == nullptr || y == nullptr)
                {
                    std::cerr << "error in triangular_matrix<..," << BE << "," << BM << ">::solve: any of the pointers is a nullptr" << std::endl;
                    return;
                }

                if (n == 0)
                {
                    return;
                }

                blas2_frame([&](const bool transpose, const T alpha, const T* x, T* y)
                {
                    // allocate local memory
                    alignas(alignment) T buffer_a[bs * bs];
                    alignas(alignment) T buffer_x[bs];

                #if defined(FP_INTEGER_GEMV)
                    alignas(alignment) T tmp_x[bs];
                    alignas(alignment) fp_type tmp_y[bs];
                #endif

                    if ((transpose && MT == matrix_type::upper_triangular) ||
                        (!transpose && MT == matrix_type::lower_triangular))
                    {
                        const std::size_t n_blocks = (n + bs - 1) / bs;
                        for (std::size_t bj = 0; bj < n_blocks; ++bj)
                        {
                            const std::size_t mm = std::min(n - bj * bs, bs);
                            for (std::size_t jj = 0; jj < bs; ++jj)
                            {
                                buffer_x[jj] = f_0;
                            }

                            for (std::size_t bi = 0; bi < bj; ++bi)
                            {
                                const std::size_t nn = std::min(n - bi * bs, bs);

                                // apply general matrix vector multiplication
                            #if defined(FP_INTEGER_GEMV)
                                if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                                {
                                    const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                    const T* fptr = reinterpret_cast<const T*>(&compressed_data[k]);
                                    const T rescale_p_3 = fptr[0];
                                    const T rescale_p_4 = fptr[1];
                                    const fp_type* tmp_a = reinterpret_cast<const fp_type*>(&fptr[2]);
                                    
                                    // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                    //
                                    // the following gemm call uses alpha=1 internally
                                    T rescale_dummy, rescale_p_1, rescale_p_2;
                                    rescale_p_1 = f_1;
                                    rescale_p_2 = f_0;
                                    for (std::size_t ii = 0; ii < nn; ++ii)
                                    {
                                        rescale_p_2 += y[bi * bs + ii];
                                    }
                                    if (transpose)
                                    {
                                        blas::gemv(L, true, nn, mm, &tmp_a[0], &y[bi * bs], &tmp_x[0]);
                                    }
                                    else
                                    {
                                        blas::gemv(L, false, mm, nn, &tmp_a[0], &y[bi * bs], &tmp_x[0]);
                                    }
                                    // ..finalize gemm call: rescaling
                                    const T a = rescale_p_4;
                                    const T b = rescale_p_2 * rescale_p_3;
                                    for (std::size_t ii = 0; ii < mm; ++ii)
                                    {
                                        buffer_x[ii] += (tmp_x[ii] * a + b);
                                    }
                                }
                                else                            
                            #endif
                                {
                                    // decompress the 'buffer'
                                    const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                    fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], mm * nn);

                                    if (transpose)
                                    {
                                        const std::size_t lda = (L == matrix_layout::rowmajor ? mm : nn);
                                        blas::gemv(cblas_layout, CblasTrans, nn, mm, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                    }
                                    else
                                    {
                                        const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                        blas::gemv(cblas_layout, CblasNoTrans, mm, nn, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                    }
                                }
                            }

                            for (std::size_t jj = 0; jj < mm; ++jj)
                            {
                                y[bj * bs + jj] = x[bj * bs + jj] - buffer_x[jj];
                            }

                            // decompress the 'buffer'
                            const std::size_t k = get_offset(MT, bj, bj);
                            fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], (mm * (mm + 1)) / 2);

                            // apply triangular solve 
                            blas::tpsv(cblas_layout, (MT == matrix_type::upper_triangular ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, mm, &buffer_a[0], &y[bj * bs], 1);
                        }
                    }
                    else if ((!transpose && MT == matrix_type::upper_triangular) ||
                            (transpose && MT == matrix_type::lower_triangular))
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

                            #if defined(FP_INTEGER_GEMV)
                                if ((BE == 0 && BM == 7) || (BE == 0 && BM == 15))
                                {
                                    const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                    const T* fptr = reinterpret_cast<const T*>(&compressed_data[k]);
                                    const T rescale_p_3 = fptr[0];
                                    const T rescale_p_4 = fptr[1];
                                    const fp_type* tmp_a = reinterpret_cast<const fp_type*>(&fptr[2]);
                                    
                                    // integer gemm : (1 x k) * (k x n) -> (1 x n)
                                    //
                                    // the following gemm call uses alpha=1 internally
                                    T rescale_dummy, rescale_p_1, rescale_p_2;
                                    rescale_p_1 = f_1;
                                    rescale_p_2 = f_0;
                                    for (std::size_t ii = 0; ii < nn; ++ii)
                                    {
                                        rescale_p_2 += y[bi * bs + ii];
                                    }
                                    if (transpose)
                                    {
                                        blas::gemv(L, true, nn, mm, &tmp_a[0], &y[bi * bs], &tmp_x[0]);
                                    }
                                    else
                                    {
                                        blas::gemv(L, false, mm, nn, &tmp_a[0], &y[bi * bs], &tmp_x[0]);
                                    }
                                    // ..finalize gemm call: rescaling
                                    const T a = rescale_p_1 * rescale_p_4;
                                    const T b = rescale_p_2 * rescale_p_3;
                                    for (std::size_t ii = 0; ii < mm; ++ii)
                                    {
                                        buffer_x[ii] += (tmp_x[ii] * a + b);
                                    }
                                }
                                else                            
                            #endif
                                {
                                    // decompress the 'buffer'
                                    const std::size_t k = (transpose ? get_offset(MT, bi, bj) : get_offset(MT, bj, bi));
                                    fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], mm * nn);

                                    // apply general matrix vector multiplication
                                    if (transpose)
                                    {
                                        const std::size_t lda = (L == matrix_layout::rowmajor ? mm : nn);
                                        blas::gemv(cblas_layout, CblasTrans, nn, mm, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                    }
                                    else
                                    {
                                        const std::size_t lda = (L == matrix_layout::rowmajor ? nn : mm);
                                        blas::gemv(cblas_layout, CblasNoTrans, mm, nn, f_1, &buffer_a[0], lda, &y[bi * bs], 1, f_1, &buffer_x[0], 1);
                                    }
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
                            fp_stream<BM, BE>::decompress(&compressed_data[k], &buffer_a[0], (mm * (mm + 1)) / 2);

                            // apply triangular solve 
                            blas::tpsv(cblas_layout, (MT == matrix_type::upper_triangular ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, mm, &buffer_a[0], &y[bj * bs], 1);

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

            void triangular_solve(const bool transpose, const T alpha, std::vector<T>& x, const std::vector<T>& y) const
            {
                triangular_solve(transpose, alpha, &x[0], &y[0]);
            }

            void tpsv(const bool transpose, const T alpha, T* x, const T* y) const
            {
                triangular_solve(transpose, alpha, x, y);
            }

            void tpsv(const bool transpose, const T alpha, std::vector<T>& x, const std::vector<T>& y) const
            {
                triangular_solve(transpose, alpha, &x[0], &y[0]);
            }
        };
    }
}

#endif