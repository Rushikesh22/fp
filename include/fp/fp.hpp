// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_HPP)
#define FP_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>
#include <assert.h>
#include <immintrin.h>

#if !defined(FP_NAMESPACE)
    #define FP_NAMESPACE fw 
#endif

namespace FP_NAMESPACE
{
    namespace internal
    {
    #if defined(__AVX512F__)
        constexpr std::size_t alignment = 64;
    #else
        constexpr std::size_t alignment = 32;
    #endif

        //! \brief Get maximum
        //!
        //! \tparam T data type
        //! \param in pointer to the input stream
        //! \param n length of the input stream
        //! \return maximum
        template <typename T>
        static inline T scan_max(const T* in, const std::size_t n)
        {
            constexpr T f_0 = static_cast<T>(0);

            if (n == 0)
            {
                return f_0;
            }

            T maximum = in[0];
            for (std::size_t i = 0; i < n; ++i)
            {
                maximum = std::max(maximum, in[i]);
            }

            return maximum;
        }

        //! \brief Get absolute maximum
        //!
        //! \tparam T data type
        //! \param in pointer to the input stream
        //! \param n length of the input stream
        //! \return absolute maximum
        template <typename T>
        static inline T scan_absmax(const T* in, const std::size_t n)
        {
            constexpr T f_0 = static_cast<T>(0);

            if (n == 0)
            {
                return f_0;
            }

            T maximum = in[0];

            for (std::size_t i = 0; i < n; ++i)
            {
                maximum = std::max(maximum, std::abs(in[i]));
            }

            return maximum;
        }

        //! \brief Get minimum
        //!
        //! \tparam T data type
        //! \param in pointer to the input stream
        //! \param n length of the input stream
        //! \return minimum
        template <typename T>
        static inline T scan_min(const T* in, const std::size_t n)
        {
            constexpr T f_0 = static_cast<T>(0);

            if (n == 0)
            {
                return f_0;
            }

            T minimum = in[0];
        
            for (std::size_t i = 0; i < n; ++i)
            {
                minimum = std::min(minimum, in[i]);
            }

            return minimum;
        }

        //! \brief Get absolute minimum
        //!
        //! \tparam T data type
        //! \param in pointer to the input stream
        //! \param n length of the input stream
        //! \return absolute minimum
        template <typename T>
        static inline T scan_absmin(const T* in, const std::size_t n)
        {
            constexpr T f_0 = static_cast<T>(0);

            if (n == 0)
            {
                return f_0;
            }

            T minimum = in[0];

            for (std::size_t i = 0; i < n; ++i)
            {
                minimum = std::min(minimum, std::abs(in[i]));
            }

            return minimum;
        }
    }

    //! \brief Bits IEEE754
    //!
    //! \tparam T floating point data type
    template <typename T>
    struct ieee754_fp;

    template <>
    struct ieee754_fp<double>
    {
        static constexpr std::uint32_t bm = 52;
        static constexpr std::uint32_t be = 11;
    };

    template <>
    struct ieee754_fp<float>
    {
        static constexpr std::uint32_t bm = 23;
        static constexpr std::uint32_t be = 8;
    };

    namespace internal
    {
        //! \brief Test for IEEE754 double type (default=false)
        //!
        //! \tparam BM bits mantissa
        //! \tparam BE bits exponent
        template <std::uint32_t BM, std::uint32_t BE>
        struct is_ieee754_double_type
        {
            static constexpr bool value = false;
        };

        template <>
        struct is_ieee754_double_type<ieee754_fp<double>::bm, ieee754_fp<double>::be>
        {
            static constexpr bool value = true;
        };

        //! \brief Test for IEEE754 single type (default=false)
        //!
        //! \tparam BM bits mantissa
        //! \tparam BE bits exponent
        template <std::uint32_t BM, std::uint32_t BE>
        struct is_ieee754_single_type
        {
            static constexpr bool value = false;
        };
        
        template <>
        struct is_ieee754_single_type<ieee754_fp<float>::bm, ieee754_fp<float>::be>
        {
            static constexpr bool value = true;
        };

        //! \brief Test for IEEE754 double or single type
        //!
        //! \tparam BM bits mantissa
        //! \tparam BE bits exponent
        template <std::uint32_t BM, std::uint32_t BE>
        struct is_ieee754_fp_type
        {
            static constexpr bool value = is_ieee754_double_type<BM, BE>::value || is_ieee754_single_type<BM, BE>::value;
        };

        //! \brief Test for bfloat16 type
        //!
        //! \tparam BM bits mantissa
        //! \tparam BE bits exponent
        template <std::uint32_t BM, std::uint32_t BE>
        struct is_bfloat16_fp_type
        {
            static constexpr bool value = false;
        };

        template <>
        struct is_bfloat16_fp_type<7, 8>
        {
            static constexpr bool value = true;
        };

        //! \brief Test for fixed point type
        //!
        //! Fixed point only if BE=0 
        //! 
        //! \tparam BM bits mantissa
        //! \tparam BE bits exponent
        template <std::uint32_t BM, std::uint32_t BE>
        struct is_fixed_point_type
        {
            static constexpr bool value = false;
        };

        template <std::uint32_t BM>
        struct is_fixed_point_type<BM, 0>
        {
            static constexpr bool value = true;
        };
    }

    //! \brief Floating / fixed point data stream
    //! 
    //! \tparam BM bits mantissa
    //! \tparam BE bits exponent
    template <std::uint32_t BM, std::uint32_t BE = 0>
    class fp_stream
    {
        static constexpr bool is_supported()
        {
            using namespace internal;

            // which BM and BE parameters are supported?
            return is_ieee754_fp_type<BM, BE>::value || (BM > 0 && BE > 0 && (BM + BE) < 16) || (BM > 0 && BM <= 16 && BE == 0);
        }

        static_assert(is_supported(), "error: unsupported <BM, BE> parameters");

        // do not allow instantiation
        fp_stream() { ; }

    public:

        static constexpr bool is_fixed_point_type = internal::is_fixed_point_type<BM, BE>::value;

        static constexpr std::uint32_t bm = BM;
        static constexpr std::uint32_t be = BE;
        static constexpr std::uint32_t bits = (is_fixed_point_type ? BM : (1 + BM + BE));

    private:

        // internal data type for the representation of the package
        using pack_t = std::uint64_t;
        static constexpr std::size_t pack_bytes = sizeof(pack_t);
        static constexpr std::size_t pack_size = (8 * pack_bytes) / bits;

    public:

        // all non-IEEE754 floating point types are represented internally through integer-typed packages:
        // packages hold as many compressed floating point numbers as possible
        using type = typename std::conditional<internal::is_ieee754_double_type<BM, BE>::value, double, 
                     typename std::conditional<internal::is_ieee754_single_type<BM, BE>::value, float,
                     typename std::conditional<internal::is_bfloat16_fp_type<BM, BE>::value, std::uint16_t, pack_t>::type>::type>::type;

        // destructor
        ~fp_stream() { ; }

        //! \brief Number of bytes needed to compress a sequence of 'n' words
        //!
        //! In case of IEEE754 floating point numbers, the number of packages is 'n'.
        //! 
        //! \param n number of floating point numbers to be compressed
        //! \return number of bytes
        static std::size_t memory_footprint_bytes(const std::size_t n)
        {
            using namespace internal;

            if (n == 0) return 0;

            if (is_ieee754_fp_type<BM, BE>::value || is_bfloat16_fp_type<BM, BE>::value)
            {
                // IEEE754 floating point numbers or bfloat16
                return n * sizeof(type);
            }
            else
            {
                // we need to store the scaling factor as well
                const std::size_t n_scaling_factor = 1;
                // number of packages to hold 'n' compressed floating point numbers
                const std::size_t num_packs = n_scaling_factor + (n + (pack_size - 1)) / pack_size;
                // number of bytes needed
                return num_packs * pack_bytes;
            }
        }

        //! \brief Number of packages needed to compress a sequence of 'n' words
        //!
        //! In case of IEEE754 floating point numbers, the number of elements (packages) is 'n'.
        //!
        //! \param n number of floating point numbers to be compressed
        //! \return number of elements (packages)
        static std::size_t memory_footprint_elements(const std::size_t n)
        {        
            using namespace internal;

            if (is_ieee754_fp_type<BM, BE>::value || is_bfloat16_fp_type<BM, BE>::value)
            {
                // IEEE754 floating point numbers or bfloat16
                return n;
            }
            else
            {
                // number of packages
                return memory_footprint_bytes(n) / pack_bytes;
            }
        }

        //! \brief Compression of floating point numbers
        //!
        //! The general idea is to truncate both the exponent (after rescaling) and the mantissa, and to pack everything into (1 + 'BE' + 'BM')-bit words
        //! which then are packed for (an almost) contiguous bit stream of compressed floating point numbers
        //!
        //! \tparam T floating point data type
        //! \param in pointer to the input sequence
        //! \param out pointer to the compressed output bit stream
        //! \param n length of the input sequence
        template <typename T>
        static void compress(const T* in, type* out, const std::size_t n)
        {
            using namespace internal;

            static_assert(std::is_floating_point<T>::value, "error: only floating numbers are allowed");
            static_assert(BE > 0, "error: cannot compress to fixed point");

            if (n == 0) return;

            if (std::is_floating_point<type>::value)
            {
                // standard floating point conversion
                if (std::is_same<T, type>::value && in == reinterpret_cast<T*>(out))
                {
                    // if pointers are the same, return immediately
                    return;
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        out[i] = in[i];
                    }
                }
            }
            else if (is_bfloat16_fp_type<BM, BE>::value)
            {
                // this special case can be handled by just storing the upper 16 bits
                if (std::is_same<T, double>::value)
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        const float ftmp = in[i];
                        const std::uint32_t itmp = *reinterpret_cast<const std::uint32_t*>(&ftmp) >> 16;
                        out[i] = itmp;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        const std::uint32_t itmp = *reinterpret_cast<const std::uint32_t*>(&in[i]) >> 16;
                        out[i] = itmp;
                    }
                }
            }
            else
            {
                // bit masks to extract IEEE754 exponent and mantissa of the single-type (float)
                constexpr std::uint32_t get_exponent = 0x7F800000U;
                constexpr std::uint32_t get_mantissa = 0x007FFFFFU;

                // minimum and maximum number of the exponent with 'BE' bits
                constexpr std::uint32_t range_min = 127 - ((0x1 << (BE - 1)) - 1);
                constexpr std::uint32_t range_max = 127 + (0x1 << (BE - 1));

                // for scaling, first determine the absolute maximum value among all uncompressed floating point numbers
                const T abs_max = scan_absmax(in, n);
                // calculate the scaling factor
                const T a = static_cast<T>(scaling_factor[BE]) / abs_max;
                // place the scaling factor as the 1st element to the output stream
                float* fptr_out = reinterpret_cast<float*>(out);
                fptr_out[0] = static_cast<float>(1.0 / a);
                // all compressed floating point numbers are placed after the scaling factor
                pack_t* ptr_out = reinterpret_cast<pack_t*>(&fptr_out[2]);

                // in case of T = 'double', there is an explicit down cast to 'float', that is, all computation below is on 32-bit words!
                std::uint32_t buffer[pack_size];
                float* fptr_buffer = reinterpret_cast<float*>(&buffer[0]);

                // process all input data in chunks of size 'package_size'
                for (std::size_t i = 0, k = 0; i < n; i += pack_size, ++k)
                {
                    // number of floating point numbers to compress / pack
                    const std::size_t ii_max = std::min(n - i, pack_size);

                    // load the floating point numbers into the local buffer and apply the scaling
                    for (std::size_t ii = 0; ii < ii_max; ++ii)
                    {
                        fptr_buffer[ii] = static_cast<float>(in[i + ii] * a);
                    }

                    // compress all 32-bit words individually: the resulting bit pattern begins at bit 0
                    for (std::size_t ii = 0; ii < ii_max; ++ii)
                    {
                        const std::uint32_t current_element = buffer[ii];
                        const std::uint32_t exponent = (current_element & get_exponent) >> ieee754_fp<float>::bm;
                        const std::uint32_t sat_exponent = std::max(std::min(exponent, range_max), range_min);
                        const std::uint32_t new_exponent = (sat_exponent - range_min) << BM;
                        const std::uint32_t new_mantissa = (current_element & get_mantissa) >> (ieee754_fp<float>::bm - BM);
                        const std::uint32_t new_sign = (current_element & 0x80000000) >> (31 - (BE + BM));

                        buffer[ii] = (new_sign | new_exponent | new_mantissa);
                    }

                    // pack the compressed floating point numbers
                    pack(buffer, ptr_out[k], ii_max);
                }
            }
        }

        //! \brief Decompression of compressed floating point numbers
        //!
        //! \tparam T floating point data type
        //! \param in pointer to the compressed input bit stream
        //! \param out pointer to the decompressed output sequence
        //! \param n length of the output sequence
        template <typename T>
        static void decompress(const type* in, T* out, const std::size_t n)
        {
            using namespace internal;

            static_assert(std::is_floating_point<T>::value, "error: only floating numbers are allowed");
            static_assert(BE > 0, "error: cannot decompress from fixed point");

            if (n == 0) return;
            
            if (std::is_floating_point<type>::value)
            {
                // standard floating point conversion
                if (std::is_same<T, type>::value && reinterpret_cast<const T*>(in) == out)
                {
                    // if pointers are the same, return immediately
                    return;
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        out[i] = in[i];
                    }
                }
            }
            else if (is_bfloat16_fp_type<BM, BE>::value)
            {
                // this special case can be handled by just recovering the upper 16 bits:
                // the lower 16 bits are zeroed
                for (std::size_t i = 0; i < n; ++i)
                {
                    const std::uint32_t tmp = static_cast<std::uint32_t>(in[i]) << 16;
                    out[i] = *reinterpret_cast<const float*>(&tmp);
                }
            }
            else
            {
                // recover the scaling factor (1st element) of the input stream
                const float* fptr_in = reinterpret_cast<const float*>(in);
                const float a = fptr_in[0];
                // move on to the packed / compressed floating point numbers
                const pack_t* ptr_in = reinterpret_cast<const pack_t*>(&fptr_in[2]); 

                // in case of T = 'double', there is an explicit up cast from 'float' to 'double', that is,
                // all computation below is on 32-bit words!
                std::uint32_t buffer[pack_size];
                const float* fptr_buffer = reinterpret_cast<const float*>(&buffer[0]);

                for (std::size_t i = 0, k = 0; i < n; i += pack_size, ++k)
                {
                    // number of floating point numbers to unpack / decompress
                    const std::size_t ii_max = std::min(n - i, pack_size);

                    // unpack the compressed floating point numbers into 'buffer'
                    unpack(ptr_in[k], buffer, ii_max);

                    // decompress all numbers individually
                    for (std::size_t ii = 0; ii < ii_max; ++ii)
                    {
                        const std::uint32_t current_element = buffer[ii];
                        const std::uint32_t exponent = (current_element & get_exponent[BE][BM]) >> BM;
                        const std::uint32_t mantissa = (current_element & get_lower_bits[BM]);
                        const std::uint32_t new_mantissa = mantissa << (31 - (ieee754_fp<float>::be + BM));
                        const std::uint32_t new_exponent = (exponent - ((0x1 << (BE - 1)) - 1) + 127) << (31 - ieee754_fp<float>::be);
                        const std::uint32_t new_sign = (buffer[ii] << (31 - (BE + BM))) & 0x80000000;

                        buffer[ii] = (new_sign | new_exponent | new_mantissa);
                    }

                    // store the floating point numbers and apply the scaling
                    for (std::size_t ii = 0; ii < ii_max; ++ii)
                    {
                        out[i + ii] = static_cast<T>(fptr_buffer[ii] * a);
                    }
                }
            }
        }
    
    private:

        // bit masks to extract the lowest [n] bits of a word with at least 16 bits
        static constexpr std::uint32_t get_lower_bits[17] =
            {0x0U, 0x1U, 0x3U, 0x7U, 0xFU, 0x1FU, 0x3FU, 0x7FU, 0xFFU, 0x1FFU, 0x3FFU, 0x7FFU, 0xFFFU, 0x1FFFU, 0x3FFFU, 0x7FFFU, 0xFFFF};

        // bit masks to extract the exponent of a compressed floating point number with ['BE']['BM'] bits
        static constexpr std::uint32_t get_exponent[17][17] = {
            {0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x0001U, 0x0002U, 0x0004U, 0x0008U, 0x0010U, 0x0020U, 0x0040U, 0x0080U, 0x0100U, 0x0200U, 0x0400U, 0x0800U, 0x1000U, 0x2000U, 0x4000U, 0x0},
            {0x0003U, 0x0006U, 0x000CU, 0x0018U, 0x0030U, 0x0060U, 0x00C0U, 0x0180U, 0x0300U, 0x0600U, 0x0C00U, 0x1800U, 0x3000U, 0x6000U, 0x0U, 0x0},
            {0x0007U, 0x000EU, 0x001CU, 0x0038U, 0x0070U, 0x00E0U, 0x01C0U, 0x0380U, 0x0700U, 0x0E00U, 0x1C00U, 0x3800U, 0x7000U, 0x0U, 0x0U, 0x0},
            {0x000FU, 0x001EU, 0x003CU, 0x0078U, 0x00F0U, 0x01E0U, 0x03C0U, 0x0780U, 0x0F00U, 0x1E00U, 0x3C00U, 0x7800U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x001FU, 0x003EU, 0x007CU, 0x00F8U, 0x01F0U, 0x03E0U, 0x07C0U, 0x0F80U, 0x1F00U, 0x3E00U, 0x7C00U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x003FU, 0x007EU, 0x00FCU, 0x01F8U, 0x03F0U, 0x07E0U, 0x0FC0U, 0x1F80U, 0x3F00U, 0x7E00U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x007FU, 0x00FEU, 0x01FCU, 0x03F8U, 0x07F0U, 0x0FE0U, 0x1FC0U, 0x3F80U, 0x7F00U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x00FFU, 0x01FEU, 0x03FCU, 0x07F8U, 0x0FF0U, 0x1FE0U, 0x3FC0U, 0x7F80U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x01FFU, 0x03FEU, 0x07FCU, 0x0FF8U, 0x1FF0U, 0x3FE0U, 0x7FC0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x03FFU, 0x07FEU, 0x0FFCU, 0x1FF8U, 0x3FF0U, 0x7FE0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x07FFU, 0x0FFEU, 0x1FFCU, 0x3FF8U, 0x7FF0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x0FFFU, 0x1FFEU, 0x3FFCU, 0x7FF8U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x1FFFU, 0x3FFEU, 0x7FFCU, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x3FFFU, 0x7FFEU, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0},
            {0x7FFFU, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0} };

        // maximum absolute floating point values that can be represented with 'BE' bits in the exponent and 16 bits total:
        // for BE = 1..8 : value = (1 - 2 ^ (BE - 16)) * 2 ^ (2 ^ (BE - 1))
        static constexpr double scaling_factor[17] = {
            1.0,
            1.999938964843750E0,
            3.999755859375000E0,
            1.599804687500000E1,
            2.559375000000000E2,
            6.550400000000000E4,
            4.290772992000000E9,
            1.841071527669059E19,
            std::numeric_limits<float>::max(),
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0 };

        //! \brief Create package of 32-bit words
        //!
        //! 'n' 32-bit words are packed into 1 package
        //!
        //! \param in pointer to the input (only the lowest 'bits' of each 32-bit word are used for the packing)
        //! \param out packed output
        //! \param n number of words to be packed
        static void pack(const std::uint32_t* in, pack_t& out, const std::size_t n)
        {
            assert(n > 0 && n <= pack_size);

            out = static_cast<pack_t>(in[0]);
            for (std::size_t i = 1; i < n; ++i)
            {
                out |= (static_cast<pack_t>(in[i]) << (i * bits));
            }
        }

        //! \brief Unpack package int 32-bit words
        //!
        //! 1 package is unpacked into 'n' 32-bit words
        //!
        //! \param in packed input
        //! \param out pointer to the unpacked output
        //! \param n number of words to be unpacked
        static void unpack(const pack_t& in, std::uint32_t* out, const std::size_t n)
        {
            assert(n > 0 && n <= pack_size);

            for (std::size_t i = 0; i < n; ++i)
            {
                out[i] = (in >> (i * bits)) & get_lower_bits[bits];
            }
        }                
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // HELPER: fixed point
    ////////////////////////////////////////////////////////////////////////////////////
    namespace internal
    {
    #if defined(__AVX2__) || defined(__AVX512F__)
        namespace 
        {
            template <typename T, typename TT>
            constexpr bool specialization_available()
            {
                constexpr bool double_or_float_to_uint8 = std::is_floating_point<T>::value && std::is_same<TT, std::uint8_t>::value;
                constexpr bool uint8_to_double_or_float = std::is_same<T, std::uint8_t>::value && std::is_floating_point<TT>::value;
                constexpr bool uint8_to_int32 = std::is_same<T, std::uint8_t>::value && std::is_same<T, std::int32_t>::value;
                constexpr bool uint8_to_int16 = std::is_same<T, std::uint8_t>::value && std::is_same<T, std::int16_t>::value;
                return double_or_float_to_uint8 || uint8_to_double_or_float || uint8_to_int32 || uint8_to_int16;
            }
        }

        //! \brief Recode between different data types using SIMD intrinsics
        //!
        //! Supported: double <-> uint8_t, uint8_t -> uint[16,32]_t
        //!
        //! \tparam T 'in' data type
        //! \tparam TT 'out' data type
        //! \param in pointer to the input sequence of type 'T'
        //! \param out pointer to the output sequence of type 'TT'
        //! \param n length of the input sequence
        //! \param a (optional) rescaling factor
        //! \param b (optional) rescaling factor
        template <typename T, typename TT, typename X = typename std::enable_if<!(specialization_available<T, TT>())>::type>
        static void recode_simd_intrinsics(const T* in, TT* out, const std::size_t n, const float a = 0.0F, const float b = 1.0F)
        {
            std::cerr << "error: recode_simd_intrinsics() is not implemented for these data types" << std::endl;
        }

        template <typename T, typename TT, typename X = typename std::enable_if<std::is_floating_point<T>::value>::type, typename Y = typename std::enable_if<std::is_same<TT, std::uint8_t>::value>::type>
        inline void recode_simd_intrinsics(const T* in, std::uint8_t* out, const std::size_t n, const float a, const float b, const X* dummy = nullptr)
        {
            if (!std::is_same<T, double>::value && !std::is_same<T, float>::value)
            {
                std::cerr << "error: floating point format is not supported" << std::endl;
                return;
            }

            // 32 8-bit words fit into an AVX2 register
            constexpr std::size_t chunk_size = 32;
            alignas(alignment) T buffer_in[chunk_size];
            alignas(alignment) std::uint8_t buffer_out[chunk_size];

            for (std::size_t i = 0; i < n; i += chunk_size)
            {
                // determine the number of elements to be compressed / packed
                const bool full_chunk = (std::min(n - i, chunk_size) == chunk_size ? true : false);
                const T* ptr_in = &in[i];
                if (!full_chunk)
                {
                    // we want to read 'chunk_size' contiguous words: load the remainder loop data into the buffer...
                    for (std::size_t ii = 0; ii < (n - i); ++ii)
                    {
                        buffer_in[ii] = in[i + ii];
                    }
                    // and switch to the buffer for reading
                    ptr_in = &buffer_in[0];
                }
                
                __m256 v256_in_1, v256_in_2, v256_in_3, v256_in_4;
                if (std::is_same<T, double>::value)
                {
                    // load 8 chunks of 4 'double' words, for a total of 32 'double's, and convert them to 'float'
                    v256_in_1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[0])))),
                                                     _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[4]))), 1);
                    v256_in_2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[8])))),
                                                     _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[12]))), 1);
                    v256_in_3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[16])))),
                                                     _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[20]))), 1);
                    v256_in_4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[24])))),
                                                     _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[28]))), 1);
                }
                else
                {
                    // load 4 chunks of 8 'float' words, for a total of 32 'float's
                    v256_in_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(&ptr_in[0]));
                    v256_in_2 = _mm256_loadu_ps(reinterpret_cast<const float*>(&ptr_in[8]));
                    v256_in_3 = _mm256_loadu_ps(reinterpret_cast<const float*>(&ptr_in[16]));
                    v256_in_4 = _mm256_loadu_ps(reinterpret_cast<const float*>(&ptr_in[24]));
                }

                // apply the rescaling so that the ouput is within the range 0.0 .. 255.0
                __m256 v256_unpacked_1 = _mm256_mul_ps(_mm256_sub_ps(v256_in_1, _mm256_set1_ps(a)), _mm256_set1_ps(b));
                __m256 v256_unpacked_2 = _mm256_mul_ps(_mm256_sub_ps(v256_in_2, _mm256_set1_ps(a)), _mm256_set1_ps(b));
                __m256 v256_unpacked_3 = _mm256_mul_ps(_mm256_sub_ps(v256_in_3, _mm256_set1_ps(a)), _mm256_set1_ps(b));
                __m256 v256_unpacked_4 = _mm256_mul_ps(_mm256_sub_ps(v256_in_4, _mm256_set1_ps(a)), _mm256_set1_ps(b));

                // convert to integer and use unsigned saturation for the packing: 32-bits -> 16-bits -> 8-bit
                __m256i v256_packedlo = _mm256_packus_epi32(_mm256_cvtps_epi32(v256_unpacked_1), _mm256_cvtps_epi32(v256_unpacked_2));
                __m256i v256_packedhi = _mm256_packus_epi32(_mm256_cvtps_epi32(v256_unpacked_3), _mm256_cvtps_epi32(v256_unpacked_4));
                // permute the output elements to recover the original order
                __m256i v256_packed = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(v256_packedlo, v256_packedhi), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

                // flush the compressed / packed data to the output
                if (full_chunk)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), v256_packed);
                }
                else
                {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(&buffer_out[0]), v256_packed);
                    for (std::size_t ii = 0; ii < (n - i); ++ii)
                    {
                        out[i + ii] = buffer_out[ii];
                    }
                }
            }
        }

        template <typename T, typename TT, typename X = typename std::enable_if<std::is_same<T, std::uint8_t>::value>::type, typename Y = typename std::enable_if<std::is_floating_point<TT>::value>::type>
        inline void recode_simd_intrinsics(const T* in, TT* out, const std::size_t n, const float a, const float b, const X* dummy = nullptr)
        {
            if (!std::is_same<TT, double>::value && !std::is_same<TT, float>::value)
            {
                std::cerr << "error: floating point format is not supported" << std::endl;
                return;
            }

            // 32 8-bit words fit into an AVX2 register
            constexpr std::size_t chunk_size = 32;
            alignas(alignment) TT buffer_out[chunk_size];

            for (std::size_t i = 0; i < n; i += chunk_size)
            {
                // load 256-bit word and permute the elements to match the reordering while unpacking: 
                // this is always safe if data allocation happened with the 'memory_footprint*' method below
                __m256i v256_packed = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[i])), _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
                // unpack 8-bit into 16-bit words
                __m256i v256_unpacked_lo = _mm256_unpacklo_epi8(v256_packed, _mm256_setzero_si256());
                __m256i v256_unpacked_hi = _mm256_unpackhi_epi8(v256_packed, _mm256_setzero_si256());

                // unpack into 32-bit words
                __m256i v256_unpacked[4];
                v256_unpacked[0] = _mm256_unpacklo_epi16(v256_unpacked_lo, _mm256_setzero_si256());
                v256_unpacked[1] = _mm256_unpackhi_epi16(v256_unpacked_lo, _mm256_setzero_si256());
                v256_unpacked[2] = _mm256_unpacklo_epi16(v256_unpacked_hi, _mm256_setzero_si256());
                v256_unpacked[3] = _mm256_unpackhi_epi16(v256_unpacked_hi, _mm256_setzero_si256());

                // determine the number of elements to be unpacked / decompressed
                const bool full_chunk = (std::min(n - i, chunk_size) == chunk_size ? true : false);
                // write the output to the buffer in case of the loop remainder
                TT* ptr_out = (full_chunk ? &out[i] : &buffer_out[0]);

                if (std::is_same<TT, double>::value)
                {
                    for (std::size_t ii = 0; ii < 4; ++ii)
                    {
                        // convert 32-bit integers to 'double' and rescale
                        __m256d tmp_1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(v256_unpacked[ii], 0)), _mm256_set1_pd(b), _mm256_set1_pd(a));
                        __m256d tmp_2 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(v256_unpacked[ii], 1)), _mm256_set1_pd(b), _mm256_set1_pd(a));
                        _mm256_storeu_pd(reinterpret_cast<double*>(&ptr_out[ii * 8 + 0]), tmp_1);
                        _mm256_storeu_pd(reinterpret_cast<double*>(&ptr_out[ii * 8 + 4]), tmp_2);
                    }
                }
                else
                {
                    for (std::size_t ii = 0; ii < 4; ++ii)
                    {
                        // convert 32-bit integers to 'float' and rescale
                        __m256 tmp = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v256_unpacked[ii]), _mm256_set1_ps(b), _mm256_set1_ps(a));
                        _mm256_storeu_ps(reinterpret_cast<float*>(&ptr_out[ii * 8]), tmp);
                    }
                }

                // flush the buffer to the output if necessary
                if (!full_chunk)
                {
                    for (std::size_t ii = 0; ii < (n - i); ++ii)
                    {
                        out[i + ii] = buffer_out[ii];
                    }
                }
            }
        }

        template <>
        inline void recode_simd_intrinsics<std::uint8_t, std::int32_t>(const std::uint8_t* in, std::int32_t* out, const std::size_t n, const float a, const float b)
        {
            // 32 8-bit words fit into an AVX2 register
            constexpr std::size_t chunk_size = 32;
            alignas(alignment) std::int32_t buffer_out[chunk_size];

            for (std::size_t i = 0; i < n; i += chunk_size)
            {
                // load 256-bit word and permute the elements to match the reordering while unpacking: 
                // this is always safe if data allocation happened with the 'memory_footprint*' method below
                __m256i v256_packed = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[i])), _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
                // unpack 8-bit into 16-bit words
                __m256i v256_unpacked_lo = _mm256_unpacklo_epi8(v256_packed, _mm256_setzero_si256());
                __m256i v256_unpacked_hi = _mm256_unpackhi_epi8(v256_packed, _mm256_setzero_si256());

                // unpack into 32-bit words
                __m256i v256_unpacked[4];
                v256_unpacked[0] = _mm256_unpacklo_epi16(v256_unpacked_lo, _mm256_setzero_si256());
                v256_unpacked[1] = _mm256_unpackhi_epi16(v256_unpacked_lo, _mm256_setzero_si256());
                v256_unpacked[2] = _mm256_unpacklo_epi16(v256_unpacked_hi, _mm256_setzero_si256());
                v256_unpacked[3] = _mm256_unpackhi_epi16(v256_unpacked_hi, _mm256_setzero_si256());

                // determine the number of elements to be unpacked / decompressed
                const bool full_chunk = (std::min(n - i, chunk_size) == chunk_size ? true : false);
                // write the output to the buffer in case of the loop remainder
                std::int32_t* ptr_out = (full_chunk ? &out[i] : &buffer_out[0]);

                // output the 32-bit integers WITHOUT converting to 'double'
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[0]), v256_unpacked[0]);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[8]), v256_unpacked[1]);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[16]), v256_unpacked[2]);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[24]), v256_unpacked[3]);

                // flush the buffer to the output if necessary
                if (!full_chunk)
                {
                    for (std::size_t ii = 0; ii < (n - i); ++ii)
                    {
                        out[i + ii] = buffer_out[ii];
                    }
                }
            }
        }
        
        template <>
        inline void recode_simd_intrinsics<std::uint8_t, std::int16_t>(const std::uint8_t* in, std::int16_t* out, const std::size_t n, const float a, const float b)
        {
            // 32 8-bit words fit into an AVX2 register
            constexpr std::size_t chunk_size = 32;
            alignas(alignment) std::int16_t buffer_out[chunk_size];

            for (std::size_t i = 0; i < n; i += chunk_size)
            {
                // load 256-bit word and permute the elements to match the reordering while unpacking: 
                // this is always safe if data allocation happened with the 'memory_footprint*' method below
                __m256i v256_packed = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[i])), _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
                // unpack 8-bit into 16-bit words
                __m256i v256_unpacked_lo = _mm256_unpacklo_epi8(v256_packed, _mm256_setzero_si256());
                __m256i v256_unpacked_hi = _mm256_unpackhi_epi8(v256_packed, _mm256_setzero_si256());

                // determine the number of elements to be unpacked / decompressed
                const bool full_chunk = (std::min(n - i, chunk_size) == chunk_size ? true : false);
                // write the output to the buffer in case of the loop remainder
                std::int16_t* ptr_out = (full_chunk ? &out[i] : &buffer_out[0]);

                // output the 16-bit integers WITHOUT any further conversion
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[0]), v256_unpacked_lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ptr_out[16]), v256_unpacked_hi);
                
                // flush the buffer to the output if necessary
                if (!full_chunk)
                {
                    for (std::size_t ii = 0; ii < (n - i); ++ii)
                    {
                        out[i + ii] = buffer_out[ii];
                    }
                }
            }
        }
        #endif
    
        //! \brief Encode floating point into fixed point numbers
        //!
        //! \tparam T floating point data type
        //! \tparam TT 'out' data type
        //! \param in pointer to the input sequence
        //! \param out pointer to the output sequence
        //! \param n length of the input sequence
        //! \param a rescaling factor
        //! \param b rescaling factor
        template <typename T, typename TT>
        static void encode_fixed_point_kernel(const T* in, TT* out, const std::size_t n, const T a, const T b)
        {
            static_assert(std::is_floating_point<T>::value, "error: only floating point numbers are allowed as input");
            static_assert(std::is_integral<TT>::value, "error: only integers are allowed as output");

            if (n == 0) return;
            
            // the scaling parameters 'a' and 'b' are stored together with the output bitstream
            float* fptr_out = reinterpret_cast<float*>(out);
            fptr_out[0] = static_cast<float>(a);
            fptr_out[1] = static_cast<float>(1.0 / b);
            TT* ptr_out = reinterpret_cast<TT*>(&fptr_out[2]);
            
        #if defined(__AVX2__) || defined(__AVX512F__)
            // use SIMD intrinsics for the recoding only in case of 8-bit fixed point representation
            constexpr bool use_simd_intrinsics = std::is_same<TT, std::uint8_t>::value;
            if (use_simd_intrinsics)
            {
                recode_simd_intrinsics<T, TT>(in, ptr_out, n, a, b);
            }
            else
        #endif
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    ptr_out[i] = (in[i] - a) * b;
                }
            }
        }

        //! \brief Encode floating point into fixed point numbers
        //!
        //! \tparam T 'in' data type
        //! \tparam TT floating point data type
        //! \param in pointer to the input sequence
        //! \param out pointer to the output sequence
        //! \param n length of the input sequence
        template <typename T, typename TT>
        static void decode_fixed_point_kernel(const T* in, TT* out, const std::size_t n)
        {
            static_assert(std::is_integral<T>::value, "error: only integers are allowed as input");
            static_assert(std::is_floating_point<TT>::value, "error: only floating point numbers are allowed as output");

            if (n == 0) return;
            
            const float* fptr_in = reinterpret_cast<const float*>(in);
            const float a = fptr_in[0];
            const float b = fptr_in[1];

        #if defined(__AVX2__) || defined(__AVX512F__)
            constexpr bool use_simd_intrinsics = std::is_same<T, std::uint8_t>::value;
            if (use_simd_intrinsics)
            {
                recode_simd_intrinsics<std::uint8_t, TT>(reinterpret_cast<const std::uint8_t*>(&fptr_in[2]), out, n, a, b);
            }
            else
        #endif
            {
                const T* ptr_in = reinterpret_cast<const T*>(&fptr_in[2]);

                for (std::size_t i = 0; i < n; ++i)
                {
                    out[i] = ptr_in[i] * b + (b * 0.5 + a);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision with 8 and 16 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <std::uint32_t BM>
    class fp_stream<BM, 0>
    {
        static_assert(BM == 8 || BM == 16, "error: only BM=8 or BM=16 is supported");

        // do not allow instantiation
        fp_stream() { ; }

    public:

        static constexpr bool is_fixed_point_type = true;

        static constexpr std::uint32_t bm = BM;
        static constexpr std::uint32_t be = 0;
        static constexpr std::uint32_t bits = BM;

        using type = typename std::conditional<(BM == 16), std::uint16_t, std::uint8_t>::type;

        //! \brief Number of bytes needed to compress a sequence of 'n' words
        //! 
        //! \param n number of floating point numbers to be compressed
        //! \return number of bytes
        static std::size_t memory_footprint_bytes(const std::size_t n)
        {
            if (n == 0) return 0;

            // we need to store the scaling factors as well
            const std::size_t n_scaling_factor = 2;
            // number of bytes needed
            return n_scaling_factor * sizeof(float) + n * sizeof(type);
        }

        //! \brief Number of elements needed to compress a sequence of 'n' words
        //!
        //! \param n number of floating point numbers to be compressed
        //! \return number of elements
        static std::size_t memory_footprint_elements(const std::size_t n)
        {
            // number of elements
            return memory_footprint_bytes(n) / sizeof(type);
        }

        //! \brief Compression of floating point numbers
        //!
        //! \tparam T floating point data type
        //! \param in pointer to the input sequence
        //! \param out pointer to the compressed output bit stream
        //! \param n length of the input sequence
        template <typename T>
        static void compress(const T* in, type* out, const std::size_t n)
        {
            using namespace internal;

            if (n == 0) return;

            // unsigned integer conversion: [minimum, maximum] -> [0, max_int]
            const T minimum = scan_min(in, n);
            const T maximum = scan_max(in, n);
        
            const T a = minimum;
            const T b = std::numeric_limits<type>::max() / (maximum - a);

            encode_fixed_point_kernel(in, out, n, a, b);
        }

        //! \brief Decompression of compressed floating point numbers
        //!
        //! \tparam T floating point data type
        //! \param in pointer to the compressed input bit stream
        //! \param out pointer to the decompressed output sequence
        //! \param n length of the output sequence
        template <typename T>
        static void decompress(const type* in, T* out, const std::size_t n)
        {
            using namespace internal;

            decode_fixed_point_kernel(in, out, n);
        }
    };

    //! \brief Definition of floating / fixed point data type
    //! 
    //! \tparam T IEEE754 double or single type
    //! \tparam BM bits mantissa
    //! \tparam BE bits exponent
    template <typename T, std::int32_t BM = ieee754_fp<T>::bm, std::uint32_t BE = ieee754_fp<T>::be>
    struct fp_type
    {
        static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' or 'float' are allowed");

        using type = T;
        static constexpr std::uint32_t bm = BM;
        static constexpr std::uint32_t be = BE;
    };

    namespace internal
    {
        //! \brief Test for 'T' being fundamental (default) or of type 'fp_type'
        //!
        //! \tparam T data type
        template <class T>
        struct is_fp_type
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct is_fp_type<fp_type<T>>
        {
            // standard 'double' and 'float' type
            static constexpr bool value = true;
        };

        template <class T>
        struct is_fp_type<fp_type<T, 7, 8>>
        {
            // truncated float type 'bfloat16'
            static constexpr bool value = true;
        };

        template <class T>
        struct is_fp_type<fp_type<T, 16, 0>>
        {
            // fixed precision 16 bit
            static constexpr bool value = true;
        };

        template <class T>
        struct is_fp_type<fp_type<T, 8, 0>>
        {
            // fixed precision 8 bit
            static constexpr bool value = true;
        };

        template <class T, class Enabled=void>
        struct extract
        {
            using type = T;
            static constexpr std::uint32_t bm = ieee754_fp<T>::bm;
            static constexpr std::uint32_t be = ieee754_fp<T>::be;
        };

        template <class T>
        struct extract<T, class std::enable_if<is_fp_type<T>::value>::type>
        {
            using type = typename T::type;
            static constexpr std::uint32_t bm = T::bm;
            static constexpr std::uint32_t be = T::be;
        };
    }
}

#endif