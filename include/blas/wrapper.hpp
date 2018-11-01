// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(WRAPPER_HPP)
#define WRAPPER_HPP

#include <cstdint>
#include <cblas.h>

#if defined(FP_USE_LIBXSMM)
#include <libxsmm.h>
#endif

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw
#endif

namespace FP_NAMESPACE
{
    namespace blas
    {
        // BLAS call wrapper: matrix matrix multiply
        //
        // helpful documentation:
        // 1) https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm

        template <typename T>
        static void gemm(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, const CBLAS_TRANSPOSE __TransB,
            const std::size_t __M, const std::size_t __N, const std::size_t __K, const T __alpha, const T* __A, const std::size_t __lda,
            const T* __B, const std::size_t __ldb,
            const T __beta, T* __C, const std::size_t __ldc);
        
        template <>
        void gemm<double>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, const CBLAS_TRANSPOSE __TransB,
            const std::size_t __M, const std::size_t __N, const std::size_t __K, const double __alpha, const double* __A, const std::size_t __lda, 
            const double* __B, const std::size_t __ldb, 
            const double __beta, double* __C, const std::size_t __ldc) 
        {
            #if defined(FP_USE_LIBXSMM)
            const char transA = (__TransA == CblasTrans ? 'T' : 'N');
            const char transB = (__TransB == CblasTrans ? 'T' : 'N');
            const libxsmm_blasint m = __M;
            const libxsmm_blasint n = __N;
            const libxsmm_blasint k = __K;
            const libxsmm_blasint lda = __lda;
            const libxsmm_blasint ldb = __ldb;
            const libxsmm_blasint ldc = __ldc;
            const double alpha = __alpha;
            const double beta = __beta;
            
            if (__Order == CblasRowMajor)
            {
                // libxsmm uses column major format per-default: A <-> B, n <-> m
                libxsmm_dgemm(&transB, &transA, &n, &m, &k, &alpha, __B, &ldb, __A, &lda, &beta, __C, &ldc);
            }
            else
            {
                libxsmm_dgemm(&transA, &transB, &m, &n, &k, &alpha, __A, &lda, __B, &ldb, &beta, __C, &ldc);
            }
            #else
            cblas_dgemm(__Order, __TransA, __TransB, __M, __N, __K, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
            #endif
        }

        template <>
        void gemm<float>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, const CBLAS_TRANSPOSE __TransB,
            const std::size_t __M, const std::size_t __N, const std::size_t __K, const float __alpha, const float* __A, const std::size_t __lda, 
            const float* __B, const std::size_t __ldb, 
            const float __beta, float* __C, const std::size_t __ldc) 
        {
            #if defined(FP_USE_LIBXSMM)
            const char transpose = 'T';
            const char notranspose = 'N';
            const char transA = (__TransA == CblasTrans ? 'T' : 'N');
            const char transB = (__TransB == CblasTrans ? 'T' : 'N');
            const libxsmm_blasint m = __M;
            const libxsmm_blasint n = __N;
            const libxsmm_blasint k = __K;
            const libxsmm_blasint lda = __lda;
            const libxsmm_blasint ldb = __ldb;
            const libxsmm_blasint ldc = __ldc;
            const float alpha = __alpha;
            const float beta = __beta;

            if (__Order == CblasRowMajor)
            {
                // libxsmm uses column major format per-default: A <-> B, n <-> m
                libxsmm_sgemm(&transB, &transA, &n, &m, &k, &alpha, __B, &ldb, __A, &lda, &beta, __C, &ldc);
            }
            else
            {
                libxsmm_sgemm(&transA, &transB, &m, &n, &k, &alpha, __A, &lda, __B, &ldb, &beta, __C, &ldc);
            }
            #else
            cblas_sgemm(__Order, __TransA, __TransB, __M, __N, __K, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
            #endif
        }

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
            #if defined(FP_GEMV_TO_GEMM)
            //const std::size_t lda = (__Order == CblasRowMajor ? __N : __M);
            const std::size_t lda = __lda;
            const std::size_t ldx = (__Order == CblasRowMajor ? __incX : (__TransA == CblasNoTrans ? __N : __M));
            const std::size_t ldy = (__Order == CblasRowMajor ? __incY : (__TransA == CblasNoTrans ? __M : __N));
            if (__TransA == CblasNoTrans)
            {
                // m, n, k : for the gemm call it is the dimensions of matrix OP(A) and OP(B) (NOT A and B)!
                gemm(__Order, __TransA, CblasNoTrans, __M, __incX, __N, __alpha, __A, lda, __X, ldx, __beta, __Y, ldy);
            }
            else
            {
                // m, n, k : for the gemm call it is the dimensions of matrix OP(A) and OP(B) (NOT A and B)!
                gemm(__Order, __TransA, CblasNoTrans, __N, __incX, __M, __alpha, __A, lda, __X, ldx, __beta, __Y, ldy);
            }
            #else
            cblas_dgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
            #endif
        }

        template <>
        void gemv<float>(const CBLAS_LAYOUT __Order, const CBLAS_TRANSPOSE __TransA, 
            const std::size_t __M, const std::size_t __N, const float __alpha, const float* __A, const std::size_t __lda, 
            const float* __X, const std::size_t __incX, 
            const float __beta, float* __Y, const std::size_t __incY) 
        {
            #if defined(FP_GEMV_TO_GEMM)
            //const std::size_t lda = (__Order == CblasRowMajor ? __N : __M);
            const std::size_t lda = __lda;
            const std::size_t ldx = (__Order == CblasRowMajor ? __incX : (__TransA == CblasNoTrans ? __N : __M));
            const std::size_t ldy = (__Order == CblasRowMajor ? __incY : (__TransA == CblasNoTrans ? __M : __N));
            if (__TransA == CblasNoTrans)
            {
                // m, n, k : for the gemm call it is the dimensions of matrix OP(A) and OP(B) (NOT A and B)!
                gemm(__Order, __TransA, CblasNoTrans, __M, __incX, __N, __alpha, __A, lda, __X, ldx, __beta, __Y, ldy);
            }
            else
            {
                // m, n, k : for the gemm call it is the dimensions of matrix OP(A) and OP(B) (NOT A and B)!
                gemm(__Order, __TransA, CblasNoTrans, __N, __incX, __M, __alpha, __A, lda, __X, ldx, __beta, __Y, ldy);
            }
            #else
            cblas_sgemv(__Order, __TransA, __M, __N, __alpha, __A, __lda, __X, __incX, __beta, __Y, __incY);
            #endif
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
