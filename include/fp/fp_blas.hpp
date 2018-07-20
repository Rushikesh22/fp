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
#include <cblas.h>

#include <fp/fp.hpp>
#include <blas/wrapper.hpp>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw
#endif

namespace FP_NAMESPACE
{
	namespace blas
	{
		//! Matrix types: lower, upper
		enum class triangular_matrix_type { lower = 0, upper = 1 };

		//! \brief General matrix
		//!
		//! \tparam T data type (of the matrix elements)
		//! \tparam BE number of bits in the exponent
		//! \tparam BM number of bits in the mantissa
		template <typename T, std::uint32_t BE = fp<T>::default_bits_exponent(), std::uint32_t BM = fp<T>::default_bits_mantissa()>
		class matrix
		{
            static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

        protected:
    
    		using fp_type = typename fp<T>::template format<BE, BM>::type;

            // extent of the matrix: 'm' rows and 'n' columns
			const std::size_t m;
			const std::size_t n;
			
            // (default) block size
			static constexpr std::size_t bs_default = 64;
            const std::size_t bs;

            // compressed matrix
            std::vector<fp_type> compressed_data;

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

            // constructor that can be called by triangular_matrix class
            matrix(const T* data, const std::size_t n, triangular_matrix_type MT = triangular_matrix_type::upper, const std::size_t bs = bs_default)
                :
                m(n),
                n(n),
                bs(bs),
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
                num_elements_c(fp<T>::template format<BE, BM>::memory_footprint_elements(bs * (n - (n / bs)))),
                num_blocks_d(((n + bs - 1) / bs) - (n / bs)),
                num_elements_d(fp<T>::template format<BE, BM>::memory_footprint_elements(((n - (n / bs)) * (n - (n / bs) + 1)) / 2)),
                num_elements(num_blocks_a * num_elements_a + num_blocks_b * num_elements_b + num_blocks_c * num_elements_c + num_blocks_d * num_elements_d)
            {
                // allocate memory for the compressed matrix
                compressed_data.reserve(num_elements);

                // compress the matrix block by block
                T buffer[bs * bs];

                std::size_t k = 0;
                for (std::size_t j = 0; j < n; j += bs)
                {
                    const std::size_t i_start = (MT == triangular_matrix_type::upper ? j : 0);
                    const std::size_t i_end = (MT == triangular_matrix_type::upper ? n : (j + 1));

                    for (std::size_t i = i_start; i < i_end; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // copy blocks into the 'buffer'
                        for (std::size_t jj = 0, kk = 0; jj < mm; ++jj)
                        {
                            // diagonal blocks
                            if (i == j)
                            {
                                const std::size_t ii_start = (MT == triangular_matrix_type::upper ? jj : 0);
                                const std::size_t ii_end = (MT == triangular_matrix_type::upper ? nn : (jj + 1));

                                for (std::size_t ii = ii_start; ii < ii_end; ++ii, ++kk)
                                {
                                    buffer[kk] = data[(j + jj) * n + (i + ii)];
                                }
                            }
                            // non-diagonal blocks
                            else
                            {
                                #pragma omp simd
                                for (std::size_t ii = 0; ii < nn; ++ii)
                                {
                                    buffer[jj * nn + ii] = data[(j + jj) * n + (i + ii)];
                                }
                            }
                        }    

                        // compress the 'buffer'
                        fp<T>::template compress<BE, BM>(&compressed_data[k], buffer, (i == j ? ((nn * (nn + 1)) / 2) : mm * nn));

                        // move on to the next block
                        if (i == j)
                        {
                            k += num_elements_a;
                        }
                        else
                        {
                            const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                            k += ((n - ij) < bs ? num_elements_c : num_elements_b);
                        }
                    }
                }
            }

            // internal methods
            virtual void apply(const bool transpose, const T alpha, const T* x, T* y) const
            {
                // allocate local memory
                std::vector<T> buffer_a;
                buffer_a.reserve(bs * bs);
                
                // apply matrix to 'x' and add the result to 'y'
                for (std::size_t j = 0, k = 0; j < m; j += bs)
                {
                    const std::size_t k_inc = ((m - j) < bs ? num_elements_c : num_elements_a);

                    for (std::size_t i = 0; i < n; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // decompress the block
                        fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                        // apply general blas matrix vector multiplication
                        gemv(CblasRowMajor, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], nn, (transpose ? &x[j] : &x[i]), 1, f_1, (transpose ? &y[i] : &y[j]), 1);
                        
                        // move on to the next block
                        k += ((n - i) < bs ? num_elements_b : k_inc);
                    }
                }
            }

		public:

			matrix() = delete;

			matrix(const T* data, const std::size_t m, const std::size_t n, const std::size_t bs = bs_default)
				:
				m(m),
				n(n),
				bs(bs),
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
                // allocate memory for the compressed matrix
                compressed_data.reserve(num_elements);

                // compress the matrix block by block
                std::vector<T> buffer;
                buffer.reserve(bs * bs);
                
                for (std::size_t j = 0, k = 0; j < m; j += bs)
                {
                    const std::size_t k_inc = ((m - j) < bs ? num_elements_c : num_elements_a);
                
                    for (std::size_t i = 0; i < n; i += bs)
                    {
                        const std::size_t mm = std::min(m - j, bs);
                        const std::size_t nn = std::min(n - i, bs);

                        // copy blocks into the 'buffer'
                        for (std::size_t jj = 0; jj < mm; ++jj)
                        {
                            #pragma omp simd
                            for (std::size_t ii = 0; ii < nn; ++ii)
                            {
                                buffer[jj * nn + ii] = data[(j + jj) * n + (i + ii)];
                            }
                        }

                        // compress the 'buffer'
                        fp<T>::template compress<BE, BM>(&compressed_data[k], &buffer[0], mm * nn);
                        
                        // move on to the next block
                        k += ((n - i) < bs ? num_elements_b : k_inc);
                    }
                }
			}

            //! \brief General matrix vector multiply
            //!
            //! Computes y = alpha * A(T) * x + beta * y
            matrix(const std::vector<T>& data, const std::size_t m, const std::size_t n, const std::size_t bs = bs_default)
                :
                matrix(&data[0], m, n, bs)
            {
                ;
            }

            std::size_t memory_footprint_bytes() const
            {
                return num_elements * sizeof(fp_type);
            }
            
            void matrix_vector(const bool transpose, const T alpha, const T* x, const T beta, T* y) const
            {
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
                std::vector<T> buffer_y;
                const bool use_buffer = (std::abs(y - x) >= std::max(m, n) ? false : true);
                buffer_y.reserve(use_buffer ? m : 0);
                T* ptr_y = nullptr;

                if (use_buffer)
                {
                    // accumulate on the buffer (do not write to the output directly)
                    ptr_y = &buffer_y[0];

                    // zero the buffer
                    #pragma omp simd
                    for (std::size_t j = 0; j < m; ++j)
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
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            ptr_y[j] = f_0;
                        }
                    }
                    else if (beta != f_1)
                    {
                        #pragma omp simd
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            ptr_y[j] *= beta;
                        }
                    }
                }
                
                apply(transpose, alpha, x, ptr_y);

                // output has been written directly
                if (!use_buffer)
                {
                    return;
                }

                // accumulate on 'y'
                if (beta == f_0)
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < m; ++j)
                    {
                        y[j] = buffer_y[j];
                    }
                }
                else if (beta == f_1)
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < m; ++j)
                    {
                        y[j] += buffer_y[j];
                    }
                }
                else
                {
                    #pragma omp simd
                    for (std::size_t j = 0; j < m; ++j)
                    {
                        y[j] = buffer_y[j] + beta * y[j];
                    }
                }
            }

            void matrix_vector(const bool transpose, const T alpha, const std::vector<T>& x, const T beta, std::vector<T>& y) const
            {
                matrix_vector(transpose, alpha, &x[0], beta, &y[0]);
            }
		};

        //! \brief Matrix
		//!
		//! \tparam T data type (of the matrix elements)
		//! \tparam BE number of bits in the exponent
		//! \tparam BM number of bits in the mantissa
		template <typename T, triangular_matrix_type MT = triangular_matrix_type::upper, std::uint32_t BE = fp<T>::default_bits_exponent(), std::uint32_t BM = fp<T>::default_bits_mantissa()>
		class triangular_matrix : public matrix<T, BE, BM>
		{
            static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

            // extent of the matrix
            using matrix<T, BE, BM>::n;

            // (default) block size
			static constexpr std::size_t bs_default = matrix<T, BE, BM>::bs_default;
            using matrix<T, BE, BM>::bs;

            // compresse matrix
            using matrix<T, BE, BM>::compressed_data;

            using matrix<T, BE, BM>::num_blocks_a;
            using matrix<T, BE, BM>::num_elements_a;

            using matrix<T, BE, BM>::num_blocks_b;
            using matrix<T, BE, BM>::num_elements_b;

            using matrix<T, BE, BM>::num_blocks_c;
            using matrix<T, BE, BM>::num_elements_c;

            using matrix<T, BE, BM>::num_blocks_d;
            using matrix<T, BE, BM>::num_elements_d;

            using matrix<T, BE, BM>::num_elements;

            // some constants
            static constexpr T f_0 = static_cast<T>(0.0);
            static constexpr T f_1 = static_cast<T>(1.0);

            // internal methods
            virtual void apply(const bool transpose, const T alpha, const T* x, T* y) const
            {
                // allocate local memory
                std::vector<T> buffer_a, buffer_y;
                buffer_a.reserve(bs * bs);
                buffer_y.reserve(bs);

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
                            tpmv(CblasRowMajor, (MT == triangular_matrix_type::upper ? CblasUpper : CblasLower), (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, nn, &buffer_a[0], &buffer_y[0], 1);

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
                            k += num_elements_a;
                        }
                        else
                        {
                            // decompress the 'buffer'
                            fp<T>::template decompress<BE, BM>(&buffer_a[0], &compressed_data[k], mm * nn);

                            // apply blas matrix vector multiplication
                            gemv(CblasRowMajor, (transpose ? CblasTrans : CblasNoTrans), mm, nn, alpha, &buffer_a[0], nn, (transpose ? &x[j] : &x[i]), 1, f_1, (transpose ? &y[i] : &y[j]), 1);

                            const std::size_t ij = (MT == triangular_matrix_type::upper ? i : j);
                            k += ((n - ij) < bs ? num_elements_c : num_elements_b);
                        }
                    }
                }
            }

        public:

            triangular_matrix() = delete;

            triangular_matrix(const T* data, const std::size_t n, const std::size_t bs = bs_default)
                :
                matrix<T, BE, BM>(data, n, MT, bs)
            {
                ;
            }

            triangular_matrix(const std::vector<T>& data, const std::size_t n, const std::size_t bs = bs_default)
                :
                triangular_matrix(&data[0], n, bs)
            {
                ;
            }
        };
	}

	/*
	namespace blas
	{
		//! Matrix types: lower, upper and full
		enum class matrix_type { lower = 0, upper = 1, full = 2 };

		//! \brief Blocked matrix
		//!
		//! \tparam Matrix_type any of 'lower', 'upper' or 'full' (=default)
		template <matrix_type Matrix_type = matrix_type::full>
		struct blocked_matrix
		{

		public:

			//! Extent of the blocks
			static constexpr std::size_t block_size = 96;

			//! \brief Determine the number of elements needed to store an 'm x n' matrix when blocked (and compressed)
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param m number of rows
			//! \param n number of columns
			//! \param bs block size (not necessarily equal to 'block_size')
			//! \return number of elements
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t num_elements(const std::size_t m, const std::size_t n, const std::size_t bs = block_size)
			{			
				// return immediately if the matrix has zero extent
				if (m == 0 || n == 0)
				{
					return 0;
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				if (m > bs || n > bs)
				{
					// if the matrix extent is larger than the block size
					//
					// a) determine the number of blocks
					const std::size_t m_blocks = (m + bs - 1) / bs;
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					//
					// * * | *     full  | 1
					// * * | *     block | 1
					// ----+--  =  ------+--
					// * * | *     22222 | 3     
					//
					// (bs = 2, in the schematic)
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_1 = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_2 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_3 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);
					// c) return the total memory foot print of the blocked matrix
					return ((m_blocks - 1) * (n_blocks - 1) * num_elements_full_block + 
						(m_blocks - 1) * num_elements_block_1 +
						(n_blocks - 1) * num_elements_block_2 +
						num_elements_block_3);
				}
				else
				{
					// the matrix extent is smaller than the block size
					return (fp<T>::template format<BE, BM>::get_memory_footprint(n * m) + sizeof(fp_t) - 1) / sizeof(fp_t);
				}
			}
			
			//! \brief Compress an 'm x n' matrix using blocking (and compression)
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param m number of rows
			//! \param n number of columns
			//! \param a matrix to be compressed
			//! \param lda leading dimension of matrix 'a'
			//! \param a_compressed compressed matrix
			//! \param bs block size (not necessarily equal to 'block_size')
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements compressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t compress(const std::size_t m, const std::size_t n, const T* a, const std::size_t lda, typename fp<T>::template format<BE, BM>::type* a_compressed, const std::size_t bs = block_size, std::vector<T>* external_buffer = nullptr)
			{  
				// return immediately if the matrix has zero extent
				if (m == 0 || n == 0)
				{
					return 0;
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				// offset for the matrix access
				std::size_t a_offset = 0;

				if (m > bs || n > bs)
				{
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks
					const std::size_t m_blocks = (m + bs - 1) / bs;
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_1 = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_2 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_3 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);

					// a scratch pad memory is needed to hold subsequent intermediate data
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve(bs * bs);

					// iterate over all blocks and compress them block-wise
					for (std::size_t bj = 0; bj < m_blocks; ++bj)
					{
						const std::size_t a_offset_inc = (bj == (m_blocks - 1) ? num_elements_block_2 : num_elements_full_block);
						for (std::size_t bi = 0; bi < n_blocks; ++bi)
						{
							const std::size_t j_start = bj * bs;
							const std::size_t j_end = std::min(m, j_start + bs);
							const std::size_t i_start = bi * bs;
							const std::size_t i_end = std::min(n, i_start + bs);

							const std::size_t mm = j_end - j_start;
							const std::size_t nn = i_end - i_start;

							// copy the current block into the scratch pad memory
							for (std::size_t j = j_start, jj = 0; j < j_end; ++j, ++jj)
							{
								for (std::size_t i = i_start, ii = 0; i < i_end; ++i, ++ii)
								{
									buffer[jj * nn + ii] = a[j * lda + i];
								}
							}

							// compress the current block and move on to the next chunk of the output buffer
							fp<T>::template compress<BE, BM>(&a_compressed[a_offset], &buffer[0], mm * nn);
							a_offset += a_offset_inc;
						}
						a_offset -= a_offset_inc;
						a_offset += num_elements_block_1;
					}
					a_offset -= num_elements_block_1;
					a_offset += num_elements_block_3;
				}
				else
				{
					// the matrix extent is smaller than the block size, however,
					// we still need to copy to the scratch pad memory, as 'lda' might
					// incorporate some padding
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve(m * n);

					for (std::size_t j = 0; j < m; ++j)
					{
						for (std::size_t i = 0; i < n; ++i)
						{
							buffer[j * n + i] = a[j * lda + i];
						}
					}
					
					// compress the matrix and move on to the next chunk of the output buffer
					fp<T>::template compress<BE, BM>(&a_compressed[a_offset], &buffer[0], n * m);
					a_offset += num_elements<T, BE, BM>(m, n, bs);
				}
				
				return a_offset;
			}
			
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t compress(const std::size_t n, const T* a, const std::size_t lda, typename fp<T>::template format<BE, BM>::type* dest, const std::size_t bs = block_size, std::vector<T>* external_buffer = nullptr)
			{
				return compress<T, BE, BM>(n, n, a, lda, dest, bs, external_buffer);
			}
			
			//! \brief Packed matrix-vector multiplication
			//!
			//! y = alpha * a(T) + beta * y
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param transpose
			//! \param bs block size the matrix was packed with
			//! \param m number of rows
			//! \param n number of columns
			//! \param alpha multiply matrix 'a' by this value
			//! \param a_compressed compressed matrix
			//! \param x input vector
			//! \param y output vector
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements decompressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t pmv(const bool transpose, const std::size_t bs, const std::size_t m, const std::size_t n, const T alpha, const typename fp<T>::template format<BE, BM>::type* a_compressed, const T* x, const T beta, T* y, std::vector<T>* external_buffer = nullptr)
			{
				constexpr T f_0 = static_cast<T>(0.0);
				constexpr T f_1 = static_cast<T>(1.0);

				// return immediately if the matrix has zero extent
				if (m == 0 || n == 0)
				{
					return 0;
				}

				// if 'alpha' equals zero, it is just moving or accumulating 'x' to 'y' 
				if (alpha == 0)
				{
					// if 'beta' equals zero the output is zero
					if (beta == 0)
					{
						for (std::size_t j = 0; j < m; ++j)
						{
							y[j] = 0;
						}
					}
					else
					{
						for (std::size_t j = 0; j < m; ++j)
						{
							y[j] = beta * y[j];
						}
					}

					return num_elements<T, BE, BM>(m, n, bs);
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				// offset for the matrix access
				std::size_t a_offset = 0;

				if (m > bs || n > bs)
				{
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks
					const std::size_t m_blocks = (m + bs - 1) / bs;
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_1 = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_2 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_block_3 = (fp<T>::template format<BE, BM>::get_memory_footprint((m - (m_blocks - 1) * bs) * (n - (n_blocks - 1) * bs)) + sizeof(fp_t) - 1) / sizeof(fp_t);
				
					// a scratch pad memory is needed to hold subsequent intermediate data
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// we need to reserve memory for both 'tmp_y' and the matrix decompression
					buffer.reserve(m + bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					T* a = &buffer[m];
					// set pointer 'tmp_y' (for the matrix-vector multiplication)
					T* y_tmp = &buffer[0];
					// initialize the output to zero
					for (std::size_t j = 0; j < m; ++j)
					{
						y_tmp[j] = 0;
					}

					// apply blocks to 'x' one after the oter
					for (std::size_t bj = 0; bj < m_blocks; ++bj)
					{
						const std::size_t a_offset_inc = (bj == (m_blocks - 1) ? num_elements_block_2 : num_elements_full_block);
						for (std::size_t bi = 0; bi < n_blocks; ++bi)
						{
							const std::size_t j_start = bj * bs;
							const std::size_t j_end = std::min(m, j_start + bs);
							const std::size_t i_start = bi * bs;
							const std::size_t i_end = std::min(n, i_start + bs);

							const std::size_t mm = j_end - j_start;
							const std::size_t nn = i_end - i_start;

							// decompress the current block
							fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], mm * nn);
							// accumulate on 'tmp_y' -> use 'beta = 1'
							if (transpose)
							{
								gemv(CblasRowMajor, CblasTrans, mm, nn, alpha, a, nn, &x[bj * bs], 1, f_1, &y_tmp[bi * bs], 1);
							}
							else
							{
								gemv(CblasRowMajor, CblasNoTrans, mm, nn, alpha, a, nn, &x[bi * bs], 1, f_1, &y_tmp[bj * bs], 1); 
							}
							a_offset += a_offset_inc;
						}	
						a_offset -= a_offset_inc;
						a_offset += num_elements_block_1;
						
					}
					a_offset -= num_elements_block_1;
					a_offset += num_elements_block_3;
				
					// in case we did not directly accumulate on 'y'
					if (beta == 0)
					{
						// just move the content pointed to by 'y_tmp' to 'y'
						for (std::size_t j = 0; j < m; ++j)
						{
							y[j] = y_tmp[j];
						}
					}
					else if (beta == 1)
					{
						// add the content pointed to by 'y_tmp' to 'y'
						for (std::size_t j = 0; j < m; ++j)
						{
							y[j] += y_tmp[j];
						}
					}
					else
					{
						// add the scaled content pointed to by 'y_tmp' to 'y'
						for (std::size_t j = 0; j < m; ++j)
						{
							y[j] += beta * y_tmp[j];
						}
					}
				}
				else
				{
					// the matrix extent is smaller than the block size, however,
					// we still need some memory for the matrix decompression
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve(m * n);
					T* a = &buffer[0];

					// decompress the matrix
					fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], m * n);
					a_offset += num_elements<T, BE, BM>(m, n, bs);

					if (transpose)
					{
						gemv(CblasRowMajor, CblasTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
					}
					else
					{
						gemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1); 
					}
				}
				
				return a_offset;
			}
		};

		//! \brief Blocked lower matrix
		template <>
		struct blocked_matrix<matrix_type::lower>
		{

		public:

			static constexpr std::size_t block_size = blocked_matrix<matrix_type::full>::block_size;

			//! \brief Determine the number of elements needed to store a lower 'n x n' matrix when blocked (and compressed)
			//!
			//! \param n number of rows and columns
			//! \return number of elements
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t num_elements(const std::size_t n, const std::size_t bs = block_size)
			{
				// return immediately if the matrix has zero extent
				if (n == 0)
				{
					return 0;
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				if (n > bs)
				{
					// if the matrix extent is larger than the block size
					//
					// a) determine the number of blocks as if the matrix would be 'full'
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_triangle_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (bs + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);
					// c) return the total memory foot print: there are 'n_blocks' triangle blocks along the diagonal
					return ((n_blocks * (n_blocks + 1) / 2 - n_blocks) * num_elements_full_block + n_blocks * num_elements_triangle_block);
				}
				else
				{
					// the matrix extent is smaller than the block size: triangle matrix
					return (fp<T>::template format<BE, BM>::get_memory_footprint(n * (n + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);
				}
			}
		};

		//! \brief Blocked upper matrix
		template <>
		struct blocked_matrix<matrix_type::upper>
		{
			static std::size_t get_offset_upper(const std::size_t j, const std::size_t i, const std::size_t n, const std::size_t d, const std::size_t f)
			{
				const std::size_t blocks_total = (n * (n + 1)) / 2;
				const std::size_t blocks_j_to_n = ((n - j) * (n - j + 1)) / 2;
				const std::size_t blocks_0_to_j = blocks_total - blocks_j_to_n;
				const std::size_t blocks_diagonal = (i > j ? j + 1 : j);
				const std::size_t blocks_i = (i - j);
				const std::size_t blocks_full = (blocks_0_to_j + blocks_i - blocks_diagonal);
				
				return (blocks_full * f + blocks_diagonal * d);
			}

		public:

			static constexpr std::size_t block_size = blocked_matrix<matrix_type::lower>::block_size;

			//! \brief Number of elements of a upper matrix
			//!
			//! \param n number of rows and columns
			//! \return number of elements
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent, std::int32_t BM = fp<T>::default_bits_mantissa>
			static std::size_t num_elements(const std::size_t n, const std::size_t bs = block_size)
			{
				return blocked_matrix<matrix_type::lower>::num_elements<T, BE, BM>(n, bs);
			}
			
			//! \brief Compress an upper 'n x n' matrix using blocking (and compression)
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param n number of rows / columns
			//! \param a matrix to be compressed
			//! \param lda leading dimension of matrix 'a'
			//! \param a_compressed compressed matrix
			//! \param bs block size (not necessarily equal to 'block_size')
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements compressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t compress(const std::size_t n, const T* a, const std::size_t lda, typename fp<T>::template format<BE, BM>::type* a_compressed, const std::size_t bs = block_size, std::vector<T>* external_buffer = nullptr)
			{
				// return immediately if the matrix has zero extent
				if (n == 0)
				{
					return 0;
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				// offset for the matrix access
				std::size_t a_offset = 0;

				if (n > bs)
				{	
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks				
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_triangle_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (bs + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);
				
					// a scratch pad memory is needed to hold subsequent intermediate data
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve(num_elements_full_block);

					// iterate over all blocks and compress them block-wise
					for (std::size_t bj = 0; bj < n_blocks; ++bj)
					{
						for (std::size_t bi = bj; bi < n_blocks; ++bi)
						{
							const std::size_t j_start = bj * bs;
							const std::size_t j_end = std::min(n, j_start + bs);
							
							if (bi == bj)
							{	
								// diagonal block
								for (std::size_t j = j_start, k = 0; j < j_end; ++j)
								{
									for (std::size_t i = j; i < j_end; ++i, ++k)
									{
										buffer[k] = a[j * lda + i];
									}
								}

								// compress upper triangle block
								const std::size_t nn = j_end - j_start;
								fp<T>::template compress<BE, BM>(&a_compressed[a_offset], &buffer[0], nn * (nn + 1) / 2);
								// move on to the next chunk of the output buffer
								a_offset += num_elements_triangle_block;
							}
							else
							{
								// non-diagonal block
								const std::size_t i_start = bi * bs;
								const std::size_t i_end = std::min(n, i_start + bs);

								const std::size_t mm = j_end - j_start;
								const std::size_t nn = i_end - i_start;

								for (std::size_t j = j_start, jj = 0; j < j_end; ++j, ++jj)
								{
									for (std::size_t i = i_start, ii = 0; i < i_end; ++i, ++ii)
									{
										buffer[jj * nn + ii] = a[j * lda + i];
									}                
								}

								// compress full block matrix
								fp<T>::template compress<BE, BM>(&a_compressed[a_offset], &buffer[0], mm * nn);
								// move on to the next chunk of the output buffer
								a_offset += num_elements_full_block;
							}
						}
					}
				}
				else
				{	
					// the matrix extent is smaller than the block size, however,
					// we still need to copy to the scratch pad memory, as 'lda' might
					// incorporate some padding
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve(n * (n + 1) / 2);

					for (std::size_t j = 0, k = 0; j < n; ++j)
					{
						for (std::size_t i = j; i < n; ++i, ++k)
						{
							buffer[k] = a[j * lda + i];
						}
					}

					// compress the upper matrix and move on to the next chunk of the output buffer
					fp<T>::template compress<BE, BM>(&a_compressed[a_offset], &buffer[0], n * (n + 1) / 2);
					a_offset += num_elements<T, BE, BM>(n, bs);
				}
				
				return a_offset;
			}
			
			//! \brief Packed matrix-vector multiplication
			//!
			//! y = alpha * a(T) + beta * y
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param transpose
			//! \param bs block size the matrix was packed with
			//! \param m number of rows
			//! \param n number of columns
			//! \param alpha multiply matrix 'a' by this value
			//! \param a_compressed compressed matrix
			//! \param x input vector
			//! \param y output vector
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements decompressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t pmv(const bool transpose, const std::size_t bs, const std::size_t n, const T alpha, const typename fp<T>::template format<BE, BM>::type* a_compressed, const T* x, const T beta, T* y, std::vector<T>* external_buffer = nullptr)
			{
				constexpr T f_0 = static_cast<T>(0.0);
				constexpr T f_1 = static_cast<T>(1.0);

				// return immediately if the matrix has zero extent
				if (n == 0)
				{
					return 0;
				}

				// distance between input and output pointer
				const std::ptrdiff_t d_xy = y - x;

				// if 'alpha' equals zero, it is just moving or accumulating 'x' to 'y' 
				if (alpha == 0)
				{
					// if 'beta' equals zero the output is zero
					if (beta == 0)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = 0;
						}
					}
					else
					{
						if (d_xy < 0)
						{
							// 'y' is located before 'x' -> forward iteration
							for (std::size_t j = 0; j < n; ++j)
							{
								y[j] = beta * x[j];
							}
						}
						else
						{
							// 'y' is located behind 'x' -> backward iteration
							for (std::size_t j = (n - 1); j > 0; --j)
							{
								y[j] = beta * x[j];
							}
							y[0] = beta * x[0];
						}
					}

					return num_elements<T, BE, BM>(n, bs);
				}
				
				// a scratch pad memory is needed to hold subsequent intermediate data
				std::vector<T> local_buffer(0);
				// use the external or the local one
				std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);

				T* a = nullptr;
				T* y_tmp = nullptr;
				if (std::abs(d_xy) >= n)
				{
					// 'x' and 'y' do not overlap, so we can accumulate on 'y' directly,
					// hence, we only need to reserve memory for the matrix decompression
					buffer.reserve(bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					a = &buffer[0];
					// we can accumulate on 'y' directly
					y_tmp = y;
				}
				else
				{
					// 'x' and 'y' overlap, so we cannot accumulate on 'y' directly,
					// hence, we need to reserve memory for both 'tmp_y' and the matrix decompression
					buffer.reserve(n + bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					a = &buffer[n];
					// set pointer 'tmp_y' (for the matrix-vector multiplication)
					y_tmp = &buffer[0];
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				// offset for the matrix access
				std::size_t a_offset = 0;

				if (n > bs)
				{
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_triangle_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (bs + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);

					// process all diagonal blocks first
					for (std::size_t bj = 0;  bj < n_blocks; ++bj)
					{

						const std::size_t j_start = bj * bs;
						const std::size_t j_end = std::min(n, j_start + bs);
						const std::size_t nn = j_end - j_start;

						// initialize 'y_tmp' with the chunk of the input 'x' being processed
						for (std::size_t jj = 0; jj < nn; ++jj)
						{
							y_tmp[bj * bs + jj] = x[bj * bs + jj];
						}

						// decompress the diagonal block (triangle)
						fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], nn * (nn + 1) / 2);
						// move on to the next diagonal block
						a_offset += (num_elements_triangle_block + (n_blocks - bj - 1) * num_elements_full_block);

						// apply triangular matrix-vector multiplication
						if (transpose)
						{
							tpmv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, nn, a, &y_tmp[bj * bs], 1);
						}
						else
						{
							tpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, nn, a, &y_tmp[bj * bs], 1);
						}
					}

					// reset the offset for the matrix access
					a_offset = 0;

					// now processes all non-diagonal blocks
					for (std::size_t bj = 0; bj < n_blocks; ++bj)
					{
						for (std::size_t bi = bj; bi < n_blocks; ++bi)
						{
							// skip diagonal blocks
							if (bi == bj)
							{
								a_offset += num_elements_triangle_block;
								continue;
							}

							const std::size_t j_start = bj * bs;
							const std::size_t j_end = std::min(n, j_start + bs);
							const std::size_t i_start = bi * bs;
							const std::size_t i_end = std::min(n, i_start + bs);

							const std::size_t mm = j_end - j_start;
							const std::size_t nn = i_end - i_start;

							// decompress full block
							fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], mm * nn);
							// and move on to the next block
							a_offset += num_elements_full_block;

							// apply matrix-vector multiplication
							if (transpose)
							{
								gemv(CblasRowMajor, CblasTrans, mm, nn, f_1, a, nn, &x[bj * bs], 1, f_1, &y_tmp[bi * bs], 1);
							}
							else
							{
								gemv(CblasRowMajor, CblasNoTrans, mm, nn, f_1, a, nn, &x[bi * bs], 1, f_1, &y_tmp[bj * bs], 1); 
							}  
						}
					}
				}
				else
				{
					// the matrix extent is smaller than the block size

					// initialize 'y_tmp' with the input vector 'x'
					for (std::size_t i = 0; i < n; ++i)
					{
						y_tmp[i] = x[i];
					}

					// decompress the matrix
					fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], n * (n + 1) / 2);
					a_offset += num_elements<T, BE, BM>(n, bs);
					
					// apply triangular matrix-vector multiplication
					if (transpose)
					{
						tpmv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, n, a, y_tmp, 1);
					}
					else
					{
						tpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, y_tmp, 1);  
					}
				}

				// in case of either the input and output vectors 'x' and 'y' overlap,
				// or 'alpha != 1' or 'beta != 0', we need to transform the result of
				// the triangular matrix-vector multiplication
				if (std::abs(d_xy) < n || beta != 0 || alpha != 1)
				{
					if (beta == 0 && alpha == 1)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = y_tmp[j];
						}
					} 
					else if (beta == 0)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = alpha * y_tmp[j];
						}
					}
					else
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = beta * y[j] + alpha * y_tmp[j];
						}
					}
				}

				return a_offset;
			}

			//! \brief Symmetric packed matrix-vector multiplication
			//!
			//! y = alpha * a(T) + beta * y
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param transpose
			//! \param bs block size the matrix was packed with
			//! \param n number of rows / columns
			//! \param alpha multiply matrix 'a' by this value
			//! \param a_compressed compressed matrix
			//! \param x input vector
			//! \param beta
			//! \param y output vector
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements decompressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t spmv(const bool transpose, const std::size_t bs, const std::size_t n, const T alpha, const typename fp<T>::template format<BE, BM>::type* a_compressed, const T* x, const T beta, T* y, std::vector<T>* external_buffer = nullptr)
			{
				constexpr T f_0 = static_cast<T>(0.0);
				constexpr T f_1 = static_cast<T>(1.0);

				// return immediately if the matrix has zero extent
				if (n == 0)
				{
					return 0;
				}

				// distance between input and output pointer
				const std::ptrdiff_t d_xy = y - x;

				// if 'alpha' equals zero, it is just moving or accumulating 'x' to 'y' 
				if (alpha == 0)
				{
					// if 'beta' equals zero the output is zero
					if (beta == 0)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = 0;
						}
					}
					else
					{
						if (d_xy < 0)
						{
							// 'y' is located before 'x' -> forward iteration
							for (std::size_t j = 0; j < n; ++j)
							{
								y[j] = beta * x[j];
							}
						}
						else
						{
							// 'y' is located behind 'x' -> backward iteration
							for (std::size_t j = (n - 1); j > 0; --j)
							{
								y[j] = beta * x[j];
							}
							y[0] = beta * x[0];
						}
					}

					return num_elements<T, BE, BM>(n, bs);
				}
				
				// a scratch pad memory is needed to hold subsequent intermediate data
				std::vector<T> local_buffer(0);
				// use the external or the local one
				std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);

				T* a = nullptr;
				T* as = nullptr;
				T* y_tmp = nullptr;
				if (std::abs(d_xy) >= n)
				{
					// 'x' and 'y' do not overlap, so we can accumulate on 'y' directly,
					// hence, we only need to reserve memory for the matrix decompression
					buffer.reserve(2 * bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					a = &buffer[0];
					as = &a[bs * (bs + 1) / 2];
					// we can accumulate on 'y' directly
					y_tmp = y;
				}
				else
				{
					// 'x' and 'y' overlap, so we cannot accumulate on 'y' directly,
					// hence, we need to reserve memory for both 'tmp_y' and the matrix decompression
					buffer.reserve(n + bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					a = &buffer[n];
					as = &a[bs * (bs + 1) / 2];
					// set pointer 'tmp_y' (for the matrix-vector multiplication)
					y_tmp = &buffer[0];
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				// offset for the matrix access
				std::size_t a_offset = 0;

				if (n > bs)
				{
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_triangle_block = (fp<T>::template format<BE, BM>::get_memory_footprint(bs * (bs + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);

					// process all diagonal blocks first
					for (std::size_t bj = 0;  bj < n_blocks; ++bj)
					{

						const std::size_t j_start = bj * bs;
						const std::size_t j_end = std::min(n, j_start + bs);
						const std::size_t nn = j_end - j_start;

						// decompress the diagonal block (triangle)
						fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], nn * (nn + 1) / 2);
						// move on to the next diagonal block
						a_offset += (num_elements_triangle_block + (n_blocks - bj - 1) * num_elements_full_block);

						// symmetrize the diagonal block
						for (std::size_t jj = 0, k = 0; jj < nn; ++jj)
						{
							for (std::size_t ii = jj; ii < nn; ++ii, ++k)
							{
								as[jj * nn + ii] = a[k];
								as[ii * nn + jj] = a[k];
							}
						}

						// apply full matrix-vector multiplication					
						if (transpose)
						{
							gemv(CblasRowMajor, CblasTrans, nn, nn, f_1, as, nn, &x[bj * bs], 1, f_0, &y_tmp[bj * bs], 1);
						}
						else
						{
							gemv(CblasRowMajor, CblasNoTrans, nn, nn, f_1, as, nn, &x[bj * bs], 1, f_0, &y_tmp[bj * bs], 1);
						}
					}

					// reset the offset for the matrix access
					a_offset = 0;

					// now processes all non-diagonal blocks
					for (std::size_t bj = 0; bj < n_blocks; ++bj)
					{
						for (std::size_t bi = bj; bi < n_blocks; ++bi)
						{
							// skip diagonal blocks
							if (bi == bj)
							{
								a_offset += num_elements_triangle_block;
								continue;
							}

							const std::size_t j_start = bj * bs;
							const std::size_t j_end = std::min(n, j_start + bs);
							const std::size_t i_start = bi * bs;
							const std::size_t i_end = std::min(n, i_start + bs);

							const std::size_t mm = j_end - j_start;
							const std::size_t nn = i_end - i_start;

							// decompress full block
							fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], mm * nn);
							// and move on to the next block
							a_offset += num_elements_full_block;

							// apply matrix-vector multiplication
							if (transpose)
							{
								gemv(CblasRowMajor, CblasTrans, mm, nn, f_1, a, nn, &x[bj * bs], 1, f_1, &y_tmp[bi * bs], 1);
								gemv(CblasRowMajor, CblasNoTrans, mm, nn, f_1, a, nn, &x[bi * bs], 1, f_1, &y_tmp[bj * bs], 1);
							}
							else
							{
								gemv(CblasRowMajor, CblasNoTrans, mm, nn, f_1, a, nn, &x[bi * bs], 1, f_1, &y_tmp[bj * bs], 1); 
								gemv(CblasRowMajor, CblasTrans, mm, nn, f_1, a, nn, &x[bj * bs], 1, f_1, &y_tmp[bi * bs], 1); 
							} 
						}
					}
				}
				else
				{
					// the matrix extent is smaller than the block size

					// initialize 'y_tmp' with the input vector 'x'
					for (std::size_t i = 0; i < n; ++i)
					{
						y_tmp[i] = x[i];
					}

					// decompress the matrix
					fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], n * (n + 1) / 2);
					a_offset += num_elements<T, BE, BM>(n, bs);

					// symmetrize the diagonal block
					for (std::size_t j = 0, k = 0; j < n; ++j)
					{
						for (std::size_t i = j; i < n; ++i, ++k)
						{
							as[j * n + i] = a[k];
							as[i * n + j] = a[k];
						}
					}
					
					// apply triangular matrix-vector multiplication
					if (transpose)
					{
						gemv(CblasRowMajor, CblasTrans, n, n, f_1, as, n, x, 1, f_0, y_tmp, 1);
					}
					else
					{
						gemv(CblasRowMajor, CblasNoTrans, n, n, f_1, as, n, x, 1, f_0, y_tmp, 1);
					}
				}

				// in case of either the input and output vectors 'x' and 'y' overlap,
				// or 'alpha != 1' or 'beta != 0', we need to transform the result of
				// the triangular matrix-vector multiplication
				if (std::abs(d_xy) < n || beta != 0 || alpha != 1)
				{
					if (beta == 0 && alpha == 1)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = y_tmp[j];
						}
					} 
					else if (beta == 0)
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = alpha * y_tmp[j];
						}
					}
					else
					{
						for (std::size_t j = 0; j < n; ++j)
						{
							y[j] = beta * y[j] + alpha * y_tmp[j];
						}
					}
				}

				return a_offset;
			}

			//! \brief Triangular solve
			//!
			//! a(T) * x = y
			//!
			//! \tparam T data type (of the matrix elements)
			//! \tparam BE number of bits in the exponent
			//! \tparam BM number of bits in the mantissa
			//! \param transpose
			//! \param bs block size the matrix was packed with
			//! \param n number of rows / columns
			//! \param a_compressed compressed matrix
			//! \param x input vector
			//! \param y output vector
			//! \param external_buffer scratch pad memory for intermediate operations
			//! \return number of elements decompressed
			template <typename T, std::int32_t BE = fp<T>::default_bits_exponent(), std::int32_t BM = fp<T>::default_bits_mantissa()>
			static std::size_t psv(const bool transpose, const std::size_t bs, const std::size_t n, const typename fp<T>::template format<BE, BM>::type* a_compressed, T* x, std::vector<T>* external_buffer = nullptr)
			{
				constexpr T f_0 = static_cast<T>(0.0);
				constexpr T f_1 = static_cast<T>(1.0);
				constexpr T f_m1 = static_cast<T>(-1.0);

				// return immediately if the matrix has zero extent
				if (n == 0)
				{
					return 0;
				}

				using fp_t = typename fp<T>::template format<BE, BM>::type;

				if (n > bs)
				{
					// if the matrix extent is larger than the block size	
					//
					// a) determine the number of blocks
					const std::size_t n_blocks = (n + bs - 1) / bs;
					// b) determine the memory foot prints of the different kinds of blocks
					const std::size_t num_elements_full_block = (fw::fp<T>::template format<BE, BM>::get_memory_footprint(bs * bs) + sizeof(fp_t) - 1) / sizeof(fp_t);
					const std::size_t num_elements_triangle_block = (fw::fp<T>::template format<BE, BM>::get_memory_footprint(bs * (bs + 1) / 2) + sizeof(fp_t) - 1) / sizeof(fp_t);

					// a scratch pad memory is needed to hold subsequent intermediate data
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// we need to reserve memory for both 'tmp_y' and the matrix decompression
					buffer.reserve(bs + bs * bs);
					// set pointer 'a' (decompressed matrix) appropriately
					T* a = &buffer[bs];
					// set pointer 'tmp_y' (for the matrix-vector multiplication)
					T* y_tmp = &buffer[0];
	
					if (transpose)
					{
						for (std::size_t j = 0; j < n_blocks; ++j)
						{
							const std::size_t j_start = j * bs;
							const std::size_t j_end = std::min(n, j_start + bs);
							const std::size_t mm = j_end - j_start;

							std::size_t a_offset = get_offset_upper(0, j, n_blocks, num_elements_triangle_block, num_elements_full_block);
							for (std::size_t i = 0; i < j; ++i)
							{
								const std::size_t i_start = i * bs;
								const std::size_t i_end = std::min(n, i_start + bs);
								const std::size_t nn = i_end - i_start;

								// decompress the block
								fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], mm * nn);
								a_offset += ((n_blocks - (i + 2)) * num_elements_full_block + num_elements_triangle_block);
								fw::blas::gemv<T>(CblasRowMajor, CblasTrans, nn, mm, f_m1, &a[0], mm, &x[i * bs], 1, f_1, &x[j * bs], 1);
							}

							a_offset += (j > 0 ? (num_elements_full_block - num_elements_triangle_block) : 0);
							fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], (mm * (mm + 1)) / 2);
							fw::blas::tpsv<T>(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, mm, &a[0], &x[j * bs], 1);
						}

					}
					else
					{
						for (std::size_t j = (n_blocks - 1); j >= 0; --j)
						{
							const std::size_t j_start = j * bs;
							const std::size_t j_end = std::min(n, j_start + bs);
							const std::size_t mm = j_end - j_start;

							const std::size_t a_offset = get_offset_upper(j, j, n_blocks, num_elements_triangle_block, num_elements_full_block);
							for (std::size_t i = (j + 1), o = (a_offset + num_elements_triangle_block); i < n_blocks; ++i, o += num_elements_full_block)
							{
								const std::size_t i_start = i * bs;
								const std::size_t i_end = std::min(n, i_start + bs);
								const std::size_t nn = i_end - i_start;

								fp<T>::template decompress<BE, BM>(a, &a_compressed[o], mm * nn);
								fw::blas::gemv<T>(CblasRowMajor, CblasNoTrans, mm, nn, f_m1, &a[0], nn, &x[i * bs], 1, f_1, &x[j * bs], 1);
							}

							fp<T>::template decompress<BE, BM>(a, &a_compressed[a_offset], (mm * (mm + 1)) / 2);
							fw::blas::tpsv<T>(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, mm, &a[0], &x[j * bs], 1);

							if (j == 0)
							{
								break;
							}
						}
					}
				}
				else
				{
					// the matrix extent is smaller than the block size, however,
					// we still need to copy to the scratch pad memory, as 'lda' might
					// incorporate some padding
					std::vector<T> local_buffer(0);
					// use the external or the local one
					std::vector<T>& buffer = (external_buffer != nullptr ? (*external_buffer) : local_buffer);
					// adapt the size of the scratch pad
					buffer.reserve((n * (n + 1)) / 2);
					T* a = &buffer[0];

					// compress the upper matrix and move on to the next chunk of the output buffer
					fp<T>::template decompress<BE, BM>(a, &a_compressed[0], (n * (n + 1)) / 2);
					fw::blas::tpsv<T>(CblasRowMajor, CblasUpper, (transpose ? CblasTrans : CblasNoTrans), CblasNonUnit, n, &a[0], &x[0], 1);
				}

				return num_elements<T, BE, BM>(n, bs);
			}
		};
	}
	*/
}

#endif
