// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(BLAS_WRAPPER_HPP)
#define BLAS_WRAPPER_HPP

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
		void gemv<double>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
			const std::size_t __M, const std::size_t __N, const double __alpha, const double* __A, const std::size_t __lda, 
			const double* __X, const std::size_t __incX, 
			const double __beta, double* __Y, const std::size_t __incY) 
		{
			cblas_dgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
		}

		template <>
		void gemv<float>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
			const std::size_t __M, const std::size_t __N, const float __alpha, const float* __A, const std::size_t __lda, 
			const float* __X, const std::size_t __incX, 
			const float __beta, float* __Y, const std::size_t __incY) 
		{
			cblas_sgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
		}

		// BLAS call wrapper: triangular packed matrix vector multiply
		template <typename T>
		static void tpmv(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
			const std::size_t __N, const T* __Ap, T* __X, const std::size_t __incX);

		template <>
		void tpmv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
			const std::size_t __N, const double* __Ap, double* __X, const std::size_t __incX)
		{
			cblas_dtpmv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
		}

		template <>
		void tpmv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
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
		void spmv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo,
			const std::size_t __N, const double __alpha, const double* __Ap,
			const double* __X, const std::size_t __incX,
			const double __beta, double* __Y, const std::size_t __incY)
		{
			cblas_dspmv(__Order, __Uplo, __N, __alpha, __Ap, __X, __incX, __beta, __Y, __incY);
		}

		template <>
		void spmv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo,
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
		void tpsv<double>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
			const std::size_t __N, const double* __Ap, double* __X, const std::size_t __incX)
		{
			cblas_dtpsv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
		}

		template <>
		void tpsv<float>(const CBLAS_LAYOUT __Order, const CBLAS_UPLO __Uplo, const CBLAS_TRANSPOSE __TransA, const CBLAS_DIAG __Diag,
			const std::size_t __N, const float* __Ap, float* __X, const std::size_t __incX)
		{
			cblas_stpsv(__Order, __Uplo, __TransA, __Diag, __N, __Ap, __X, __incX);
		}
        }
}

#endif