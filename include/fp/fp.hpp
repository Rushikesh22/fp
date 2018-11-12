// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_HPP)
#define FP_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <immintrin.h>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw 
#endif

namespace FP_NAMESPACE
{
    //! \brief Get maximum
    //!
    //! \tparam T data type
    //! \param in pointer to the input stream
    //! \param n length of the input stream
    //! \return maximum
    template <typename T>
    static inline T scan_max(const T* in, const std::size_t n)
    {
        if (n == 0)
        {
            return 0;
        }

        T maximum = in[0];
        #pragma omp simd reduction(max : maximum)
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
        if (n == 0)
        {
            return 0;
        }

        T maximum = in[0];

        #pragma omp simd reduction(max : maximum)
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
        if (n == 0)
        {
            return 0;
        }

        T minimum = in[0];
    
        #pragma omp simd reduction(min : minimum)
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
        if (n == 0)
        {
            return 0;
        }

        T minimum = in[0];

        #pragma omp simd reduction(min : minimum)
        for (std::size_t i = 0; i < n; ++i)
        {
            minimum = std::min(minimum, std::abs(in[i]));
        }

        return minimum;
    }

    //! \brief Compressed floating / fixed point numbers
    //!
    //! \tparam floating point data type being compressed
    template <typename T>
    class fp
    {
        // what is supported? only 'T' = 'double' or 'float'
        static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

        // what is supported? compressed floating / fixed point representations with at least 2 and at most 16 bits
        static constexpr std::uint32_t min_bits_compression = 2;
        static constexpr std::uint32_t max_bits_compression = 16;

        //! \brief Packing and unpacking of words with 'TB' bits
        //! 
        //! \tparam TB total number of bits (usually, sign + exponent + mantissa)
        template <std::uint32_t TB>
        struct implementation
        {
            // what is supported?
            static_assert(TB >= min_bits_compression && TB <= max_bits_compression, "error: invalid number of bits for the compressed floating or fixed point representation");

            // 'pack_t' is the data type that is used to hold multiple compressed words 
            using pack_t = std::uint64_t;

            // size of the 'pack_t' data type in bytes
            static constexpr std::size_t pack_bytes = sizeof(pack_t);

            // number of compressed words that fit into the 'pack_t' data type
            static constexpr std::size_t pack_size = (8 * pack_bytes) / TB;

            //! \brief Number of bytes needed to compress a sequence of 'n' words with 'TB' bits each
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of bytes
            static std::size_t memory_footprint_bytes(const std::size_t n)
            {
                if (n == 0)
                {
                    return 0;
                }
                else
                {
                    const std::size_t num_packs = (n + pack_size - 1) / pack_size;
                    return num_packs * pack_bytes;
                }
            }

            //! \brief Number of elements of the 'pack_t' type needed to compress a sequence of 'n' words with 'TB' bits each
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of elements of the 'pack_t' type
            static std::size_t memory_footprint_elements(const std::size_t n)
            {
                return memory_footprint_bytes(n) / pack_bytes;
            }

            //! \brief Pack 32-bit words
            //!
            //! 'n' 32-bit words are packed into 1 word of type 'pack_t'.
            //!
            //! \param in pointer to the input (only the lowest 'TB' bits of each 32-bit word are used for the packing)
            //! \param out packed output
            //! \param n number of words to be packed
            static void pack(const std::uint32_t* in, pack_t& out, const std::size_t n)
            {
                out = static_cast<pack_t>(in[0]);
                for (std::size_t i = 1; i < n; ++i)
                {
                    out |= (static_cast<pack_t>(in[i]) << (i * TB));
                }
            }

            //! \brief Unpack 'TB'-bit words
            //!
            //! 1 word of type 'pack_t' is unpacked into 'n' 32-bit words
            //!
            //! \param in packed input
            //! \param out pointer to the unpacked output
            //! \param n number of words to be unpacked
            static void unpack(const pack_t& in, std::uint32_t* out, const std::size_t n)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    out[i] = (in >> (i * TB)) & get_lower_bits[TB];
                }
            }
        };

        //! \brief Test for exponent and mantissa bits both being larger than 0, and that they add up to at most 15
        //!
        //! \param be bits exponent
        //! \param bm bits mantissa
        //! \return condition is satisfied or not
        static constexpr bool default_case(const std::uint32_t be, const std::uint32_t bm)
        {
            return (be > 0) && (bm > 0) && ((be + bm) < max_bits_compression);
        }

        //! \brief Test for special values of exponent and mantissa bits that are defined outside the class
        //!
        //! \param be bits exponent
        //! \param bm bits mantissa
        //! \return special case or not
        static constexpr bool special_case(const std::uint32_t be, const std::uint32_t bm)
        {
            return (be == 8 && bm == 7) || (be == 0 && (bm == 7 || bm == 15));
        }

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
                {0x7FFFU, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0}};

        // bit masks to extract the lowest [n] bits of a word with at least 16 bits
        static constexpr std::uint32_t get_lower_bits[17] =
                {0x0U, 0x1U, 0x3U, 0x7U, 0xFU, 0x1FU, 0x3FU, 0x7FU, 0xFFU, 0x1FFU, 0x3FFU, 0x7FFU, 0xFFFU, 0x1FFFU, 0x3FFFU, 0x7FFFU, 0xFFFF};

        // maximum absolute floating point values that can be represented with 'BE' bits in the exponent and 16 bits total:
        // for BE = 1..8 : value = (1 - 2 ^ (15 - BE)) * 2 ^ (2 ^ (BE - 1))
        static constexpr float scaling_factor[17] = {
                1.0F,
                1.9998779296875000E0F,
                3.9995117187500000E0F,
                1.5996093750000000E1F,
                2.5587500000000000E2F,
                6.5472000000000000E4F,
                4.2865786880000000E9F,
                1.8374686479671624E19F,
                8.5070591730234616E37F,
                1.0F,
                1.0F,
                1.0F,
                1.0F,
                1.0F,
                1.0F,
                1.0F,
                1.0F};

    public:

        // default number of bits in the exponent and mantissa (according to IEEE-754)
        static constexpr std::uint32_t default_bits_exponent();
        static constexpr std::uint32_t default_bits_mantissa();

        //! \brief Compressed floating point representation with 'BE' exponent and 'BM' mantissa bits
        //!
        //! The sign bit is always implicit.
        //! Fixed point representations are those with 'BE' = 0.
        //!
        //! \tparam BE bits exponent
        //! \tparam BM bits mantissa
        template <std::uint32_t BE, std::uint32_t BM>
        struct format
        {
            static_assert(default_case(BE, BM) || special_case(BE, BM), "error: only floating point representations with 3-16 bits, and fixed point representations with 8 and 16 bits supported");

            static constexpr bool is_fixed_point = false;

            // data type for the internal representation of the compressed floating point numbers
            using type = typename implementation<1 + BE + BM>::pack_t;

            //! \brief Number of bytes needed to compress a sequence of 'n' floating point numbers
            //!
            //! \param n number of floating point number to be compressed
            //! \return number of bytes
            static std::size_t memory_footprint_bytes(const std::size_t n)
            {
                if (n == 0)
                {
                    return 0;
                }

                // floating point numbers are rescaled so as to match the output range: we need to store the scaling factor as well
                const std::size_t n_scaling_factor = (sizeof(type) * 8 + (1 + BE + BM - 1)) / (1 + BE + BM);
                return implementation<1 + BE + BM>::memory_footprint_bytes(n_scaling_factor + n);
            }

            //! \brief Number of elements of the internal data type needed to compress a sequence of 'n' floating point numbers
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of elements of the internal data type
            static std::size_t memory_footprint_elements(const std::size_t n)
            {
                return memory_footprint_bytes(n) / sizeof(type);
            }
        };        

        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<!(default_case(BE, BM) && !special_case(BE, BM))>::type>
        static void compress(const T* in, typename format<BE, BM>::type* out, const std::size_t n);
        
        //! \brief Compression (default case)
        //!
        //! The general idea is to truncate both the exponent (after rescaling) and the mantissa, and to pack everything into (1 + 'BE' + 'BM')-bit words
        //! which then are packed for (an almost) contiguous bit stream of compressed floating point numbers
        //!
        //! \tparam BE bits exponent
        //! \tparam BE bits mantissa
        //! \param in pointer to the input sequence of IEEE-754 floating point numbers
        //! \param out pointer to the compressed output bit stream
        //! \param n length of the input sequence
        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<default_case(BE, BM) && !special_case(BE, BM)>::type>
        static void compress(const T* in, typename format<BE, BM>::type* out, const std::size_t n, const X* ptr = nullptr)
        {
            if (n == 0)
            {
                return;
            }

            using fp_type = typename format<BE, BM>::type;

            // bit masks to extract IEEE-754 exponent and mantissa of the single-type (float)
            constexpr std::uint32_t get_exponent = 0x7F800000U;
            constexpr std::uint32_t get_mantissa = 0x007FFFFFU;

            // minimum and maximum number of the exponent with 'BE' bits
            constexpr std::uint32_t range_min = 127 - ((0x1 << (BE - 1)) - 1);
            constexpr std::uint32_t range_max = 127 + (0x1 << (BE - 1));

            // for rescaling, first determine the absolute maximum value among all uncompressed floating point numbers...
            const float abs_max = static_cast<float>(scan_absmax(in, n));
            // calculate the rescaling factor...
            const float a = 0.98F * scaling_factor[BE] / abs_max;
            // and place it as the 1st element to the output stream
            float* fptr_out = reinterpret_cast<float*>(out);
            fptr_out[0] = 1.0F / a;
            // all compressed floating point numbers are placed after the rescaling factor
            fp_type* ptr_out = reinterpret_cast<fp_type*>(&fptr_out[1]);

            constexpr std::size_t pack_bytes = implementation<1 + BE + BM>::pack_bytes;
            constexpr std::size_t pack_size = implementation<1 + BE + BM>::pack_size;

            // in case of T = 'double', there is an explicit down cast to 'float', that is,
            // all computation below is on 32-bit words!
            std::uint32_t buffer[pack_size];
            float* fptr_buffer = reinterpret_cast<float*>(&buffer[0]);

            // process all input data in chunks of size 'pack_size'
            for (std::size_t i = 0, k = 0; i < n; i += pack_size, ++k)
            {
                // number of elements to compress / pack
                const std::size_t ii_max = std::min(n - i, pack_size);

                // load the floating point numbers into the local buffer and apply the rescaling
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    fptr_buffer[ii] = static_cast<float>(in[i + ii]) * a;
                }

                // compress all 32-bit words individually: the resulting bit pattern begins at bit 0
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    const std::uint32_t current_element = buffer[ii];
                    const std::uint32_t exponent = (current_element & get_exponent) >> 23;
                    const std::uint32_t sat_exponent = std::max(std::min(exponent, range_max), range_min);
                    const std::uint32_t new_exponent = (sat_exponent - range_min) << BM;
                    const std::uint32_t new_mantissa = (current_element & get_mantissa) >> (23 - BM);
                    const std::uint32_t new_sign = (current_element & 0x80000000) >> (31 - (BE + BM));

                    buffer[ii] = (new_sign | new_exponent | new_mantissa);
                }

                // pack the compressed floating point numbers
                implementation<1 + BE + BM>::pack(buffer, ptr_out[k], ii_max);
            }
        }

        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<!(default_case(BE, BM) && !special_case(BE, BM))>::type>
        static void decompress(const typename format<BE, BM>::type* in, T* out, const std::size_t n);

        //! \brief Decompression (default case)
        //!
        //! \tparam BE bits exponent
        //! \tparam BE bits mantissa
        //! \param in pointer to the compressed input bit stream
        //! \param out pointer to the decompressed output sequence of IEEE-754 floating point numbers
        //! \param n length of the output sequence
        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<default_case(BE, BM) && !special_case(BE, BM)>::type>
        static void decompress(const typename format<BE, BM>::type* in, T* out, const std::size_t n, const X* ptr = nullptr)
        {
            if (n == 0)
            {
                return;
            }

            using fp_type = typename format<BE, BM>::type;

            // recover the scaling factor (1st element) of the input stream...
            const float* fptr_in = reinterpret_cast<const float*>(in);
            const float a = fptr_in[0];
            // and move on to the packed / compressed floating point numbers
            const fp_type* ptr_in = reinterpret_cast<const fp_type*>(&fptr_in[1]); 

            constexpr std::size_t pack_bytes = implementation<1 + BE + BM>::pack_bytes;
            constexpr std::size_t pack_size = implementation<1 + BE + BM>::pack_size;

            // in case of T = 'double', there is an explicit up cast from 'float' to 'double', that is,
            // all computation below is on 32-bit words!
            std::uint32_t buffer[pack_size];
            const float* fptr_buffer = reinterpret_cast<const float*>(&buffer[0]);

            for (std::size_t i = 0, k = 0; i < n; i += pack_size, ++k)
            {
                // number of elements to unpack / decompress
                const std::size_t ii_max = std::min(n - i, pack_size);

                // unpack the compressed floating point numbers into 'buffer'
                implementation<1 + BE + BM>::unpack(ptr_in[k], buffer, ii_max);

                // decompress all numbers individually
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    const std::uint32_t current_element = buffer[ii];
                    const std::uint32_t exponent = (current_element & get_exponent[BE][BM]) >> BM;
                    const std::uint32_t mantissa = (current_element & get_lower_bits[BM]);
                    const std::uint32_t new_mantissa = mantissa << (31 - (8 + BM));
                    const std::uint32_t new_exponent = (exponent - ((0x1 << (BE - 1)) - 1) + 127) << (31 - 8);
                    const std::uint32_t new_sign = (buffer[ii] << (31 - (BE + BM))) & 0x80000000;

                    buffer[ii] = (new_sign | new_exponent | new_mantissa);
                }

                // store the floating point numbers and apply the rescaling
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    out[i + ii] = static_cast<T>(fptr_buffer[ii] * a);
                }
            }
        }
    };

    template <>
    constexpr std::uint32_t fp<double>::default_bits_exponent()
    {
        return 11;
    }

    template <>
    constexpr std::uint32_t fp<double>::default_bits_mantissa()
    {
        return 52;
    }

    template <>
    constexpr std::uint32_t fp<float>::default_bits_exponent()
    {
        return 8;
    }

    template <>
    constexpr std::uint32_t fp<float>::default_bits_mantissa()
    {
        return 23;
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // NO COMPRESSION: IEEE-754 double type
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<11, 52>
    {
        // we need an integer data type that is of the same size as 'double'!
        // note: we do not use 'double' here, as we need to be able to identify the compressed type elsewhere
        using type = std::uint64_t;

        static constexpr bool is_fixed_point = false;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (n * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<11, 52>(const double* in, typename fp<double>::format<11, 52>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'double'
        double* ptr_out = reinterpret_cast<double*>(out);

        if (in == ptr_out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = in[i];
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<11, 52>(const typename fp<double>::format<11, 52>::type* in, double* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'double'
        const double* ptr_in = reinterpret_cast<const double*>(in);

        if (ptr_in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i];
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // NO COMPRESSION: IEEE-754 single type
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::format<8, 23>
    {
        // we need an integer data type that is of the same size as 'float'!
        // note: we do not use 'float' here, as we need to be able to identify the compressed type elsewhere
        using type = std::uint32_t;

        static constexpr bool is_fixed_point = false;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (n * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<8, 23>(const float* in, typename fp<float>::format<8, 23>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'float'
        float* ptr_out = reinterpret_cast<float*>(out);

        if (in == ptr_out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = in[i];
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<8, 23>(const typename fp<float>::format<8, 23>::type* in, float* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'float'
        const float* ptr_in = reinterpret_cast<const float*>(in);

        if (ptr_in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i];
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 32 bits: 8 bit exponent and 23 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<8, 23>
    {
        // we need an integer data type that is of the same size as 'float'!
        // note: we do not use 'float' here, as we need to be able to identify the compressed type elsewhere
        using type = std::uint32_t;

        static constexpr bool is_fixed_point = false;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (n * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<8, 23>(const double* in, typename fp<double>::format<8, 23>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'float'
        float* ptr_out = reinterpret_cast<float*>(out);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = static_cast<float>(in[i]);
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 23>(const typename fp<double>::format<8, 23>::type* in, double* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // switch back to 'float'
        const float* ptr_in = reinterpret_cast<const float*>(in);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<double>(ptr_in[i]);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 16 bits: 8 bit exponent and 7 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<8, 7>
    {
        using type = std::uint16_t;

        static constexpr bool is_fixed_point = false;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (n * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<8, 7>(const double* in, typename format<8, 7>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            // down cast double -> float
            const float f_tmp = in[i];
            // store the upper 16 bits: full exponent and 7 bits of the mantissa
            const std::uint32_t i_tmp = *reinterpret_cast<const std::uint32_t*>(&f_tmp) >> 16;
            out[i] = i_tmp;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 7>(const typename format<8, 7>::type* in, double* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            // store back to the upper 16 bits: the lower 16 bits are zero
            const std::uint32_t tmp = static_cast<std::uint32_t>(in[i]) << 16;
            // up cast float -> double
            out[i] = static_cast<double>(*reinterpret_cast<const float*>(&tmp));
        }
    }
    
    template <>
    template <>
    struct fp<float>::format<8, 7>
    {
        using type = std::uint16_t;

        static constexpr bool is_fixed_point = false;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (n * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<8, 7>(const float* in, typename format<8, 7>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            // store the upper 16 bits: full exponent and 7 bits of the mantissa
            const std::uint32_t tmp = *reinterpret_cast<const std::uint32_t*>(&in[i]) >> 16;
            out[i] = tmp;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<8, 7>(const typename format<8, 7>::type* in, float* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            // store back to the upper 16 bits: the lower 16 bits are zero
            const std::uint32_t tmp = static_cast<std::uint32_t>(in[i]) << 16;
            out[i] = *reinterpret_cast<const float*>(&tmp);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // HELPER: fixed precision
    ////////////////////////////////////////////////////////////////////////////////////
    #if defined(__AVX2__) || defined(__AVX512F__)
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
    template <typename T, typename TT>
    static void recode_simd_intrinsics(const T* in, TT* out, const std::size_t n, const double a = 0.0, const double b = 1.0)
    {
        std::cerr << "error: recode_simd_intrinsics() is not implemented for these data types" << std::endl;
    }

    template <>
    inline void recode_simd_intrinsics<double, std::uint8_t>(const double* in, std::uint8_t* out, const std::size_t n, const double a, const double b)
    {
        // 32 8-bit words fit into an AVX2 register
        constexpr std::size_t chunk_size = 32;
        alignas(32) double buffer_in[chunk_size];
        alignas(32) std::uint8_t buffer_out[chunk_size];

        for (std::size_t i = 0; i < n; i += chunk_size)
        {
            // determine the number of elements to be compressed / packed
            const bool full_chunk = (std::min(n - i, chunk_size) == chunk_size ? true : false);
            const double* ptr_in = &in[i];
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
            
            // load 8 chunks of 4 'double' words, for a total of 32 'double's, and convert them to 'float'
            __m256 v256_in_1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[0])))),
                                                    _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[4]))), 1);
            __m256 v256_in_2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[8])))),
                                                    _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[12]))), 1);
            __m256 v256_in_3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[16])))),
                                                    _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[20]))), 1);
            __m256 v256_in_4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[24])))),
                                                    _mm256_cvtpd_ps(_mm256_loadu_pd(reinterpret_cast<const double*>(&ptr_in[28]))), 1);

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

    template <>
    inline void recode_simd_intrinsics<std::uint8_t, double>(const std::uint8_t* in, double* out, const std::size_t n, const double a, const double b)
    {
        // 32 8-bit words fit into an AVX2 register
        constexpr std::size_t chunk_size = 32;
        alignas(32) double buffer_out[chunk_size];

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
            double* ptr_out = (full_chunk ? &out[i] : &buffer_out[0]);

            for (std::size_t ii = 0; ii < 4; ++ii)
            {
                // convert 32-bit integers to 'double' and rescale
                __m256d tmp_1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(v256_unpacked[ii], 0)), _mm256_set1_pd(b), _mm256_set1_pd(a));
                __m256d tmp_2 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(v256_unpacked[ii], 1)), _mm256_set1_pd(b), _mm256_set1_pd(a));
                _mm256_storeu_pd(reinterpret_cast<double*>(&ptr_out[ii * 8 + 0]), tmp_1);
                _mm256_storeu_pd(reinterpret_cast<double*>(&ptr_out[ii * 8 + 4]), tmp_2);
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
    inline void recode_simd_intrinsics<std::uint8_t, std::int32_t>(const std::uint8_t* in, std::int32_t* out, const std::size_t n, const double a, const double b)
    {
        // 32 8-bit words fit into an AVX2 register
        constexpr std::size_t chunk_size = 32;
        alignas(32) std::int32_t buffer_out[chunk_size];

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
    inline void recode_simd_intrinsics<std::uint8_t, std::int16_t>(const std::uint8_t* in, std::int16_t* out, const std::size_t n, const double a, const double b)
    {
        // 32 8-bit words fit into an AVX2 register
        constexpr std::size_t chunk_size = 32;
        alignas(32) std::int16_t buffer_out[chunk_size];

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

    //! \brief Recode floating point to fixed point numbers
    //!
    //! \tparam T floating point data type
    //! \tparam TT 'out' data type
    //! \param in pointer to the input sequence
    //! \param out pointer to the output sequence of type 'TT'
    //! \param n length of the input sequence
    //! \param a rescaling factor
    //! \param b rescaling factor
    template <typename T, typename TT>
    static void encode_fixed_point_kernel(const T* in, TT* out, const std::size_t n, const T a, const T b)
    {
        static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "error: only 'double' and 'float' type is supported");

        if (n == 0)
        {
            return;
        }

        TT* ptr_out = out;
        // the scaling parameters 'a' and 'b' are stored together with the output bitstream
        T* fptr_out = reinterpret_cast<T*>(out);
        fptr_out[0] = a;
        fptr_out[1] = static_cast<T>(1.0) / b;
        ptr_out = reinterpret_cast<TT*>(&fptr_out[2]);
        
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
            #pragma omp simd
            for (std::size_t i = 0; i < n; ++i)
            {
                ptr_out[i] = (in[i] - a) * b;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 16 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 15>
    {
        using type = std::uint16_t;

        static constexpr bool is_fixed_point = true;

        static constexpr std::uint16_t max_int = 0xFFFFU;
    
        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            if (n == 0)
            {
                return 0;
            }
            else
            {
                // we need to store the scaling factors as well
                return (2 * sizeof(double) + (n * sizeof(type)));
            }
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    struct fp<float>::format<0, 15>
    {
        using type = std::uint16_t;

        static constexpr bool is_fixed_point = true;

        static constexpr std::uint16_t max_int = 0xFFFFU;
     
        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            if (n == 0)
            {
                return 0;
            }
            else
            {
                // we need to store the scaling factors as well
                return (2 * sizeof(float) + (n * sizeof(type)));
            }
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 8 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 7>
    {
        using type = std::uint8_t;

        static constexpr bool is_fixed_point = true;

        static constexpr std::uint8_t max_int = 0xFFU;
     
        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            if (n == 0)
            {
                return 0;
            }
            else
            {
                // we need to store the scaling factors as well
                #if defined(__AVX2__) || defined(__AVX512F__)
                // in this case, the SIMD intrinsics scheme is used for the recoding, and
                // we have to make sure that the data being processed (32 8-bit words) 
                // can be accessed through SIMD loads
                constexpr std::size_t chunk_size = 32;
                return (2 * sizeof(double) + (((n + (chunk_size - 1)) / chunk_size) * chunk_size * sizeof(type)));
                #else
                return (2 * sizeof(double) + (n * sizeof(type)));
                #endif
            }
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    struct fp<float>::format<0, 7>
    {
        using type = std::uint8_t;

        static constexpr bool is_fixed_point = true;

        static constexpr std::uint8_t max_int = 0xFFU;
     
        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            if (n == 0)
            {
                return 0;
            }
            else
            {
                // we need to store the scaling factors as well
                return (2 * sizeof(float) + (n * sizeof(type)));
            }
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    //! \brief Recode floating point to 16-bit fixed point numbers
    //!
    //! \tparam T floating point data type
    //! \param in pointer to the input sequence
    //! \param out pointer to the 16-bit fixed point output sequence
    //! \param n length of the input sequence
    template <typename T>
    static void encode_fixed_point(const T* in, typename fp<T>::template format<0, 15>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        // unsigned integer conversion: [minimum, maximum] -> [0, 0xFFFFU]
        const T minimum = scan_min(in, n);
        const T maximum = scan_max(in, n);

        const T a = minimum;
        const T b = fp<T>::template format<0, 15>::max_int / (maximum - a);
    
        encode_fixed_point_kernel(in, out, n, a, b);
    }

    //! \brief Recode floating point to 16-bit fixed point numbers
    //!
    //! \tparam T floating point data type
    //! \param in pointer to the input sequence
    //! \param out pointer to the 16-bit fixed point output sequence
    //! \param n length of the input sequence
    template <typename T>
    static void encode_fixed_point(const T* in, typename fp<T>::template format<0, 7>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
    
        // unsigned integer conversion: [minimum, maximum] -> [0, 0xFFU]
        const T minimum = scan_min(in, n);
        const T maximum = scan_max(in, n);
    
        const T a = minimum;
        const T b = fp<T>::template format<0, 7>::max_int / (maximum - a);
    
        encode_fixed_point_kernel(in, out, n, a, b);
    }
    
    template <typename T>
    static void decode_fixed_point(const typename fp<T>::template format<0, 15>::type* in, T* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        const T* fptr_in = reinterpret_cast<const T*>(in);
        const T a = fptr_in[0];
        const T b = fptr_in[1];

        using in_t = typename fp<T>::template format<0, 15>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i] * b + a;
        }
    }

    template <typename T>
    static void decode_fixed_point(const typename fp<T>::template format<0, 7>::type* in, T* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        const T* fptr_in = reinterpret_cast<const T*>(in);
        const T a = fptr_in[0];
        const T b = fptr_in[1];

        using in_t = typename fp<T>::template format<0, 7>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #if defined(__AVX2__) || defined(__AVX512F__)
        constexpr bool use_simd_intrinsics = std::is_same<typename fp<T>::template format<0, 7>::type, std::uint8_t>::value;
        if (use_simd_intrinsics)
        {
            recode_simd_intrinsics<std::uint8_t, T>(ptr_in, out, n, a, b);
        }
        else
        #endif
        {
            #pragma omp simd
            for (std::size_t i = 0; i < n; ++i)
            {
                out[i] = ptr_in[i] * b + a;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 16 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    inline void fp<double>::compress<0, 15>(const double* in, typename format<0, 15>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
                
        encode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 15>(const typename format<0, 15>::type* in, double* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
        
        decode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<float>::compress<0, 15>(const float* in, typename format<0, 15>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
        
        encode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 15>(const typename format<0, 15>::type* in, float* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
        
        decode_fixed_point(in, out, n);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 8 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    inline void fp<double>::compress<0, 7>(const double* in, typename format<0, 7>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        encode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 7>(const typename format<0, 7>::type* in, double* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
        
        decode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<float>::compress<0, 7>(const float* in, typename format<0, 7>::type* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }

        encode_fixed_point(in, out, n);
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 7>(const typename format<0, 7>::type* in, float* out, const std::size_t n)
    {
        if (n == 0)
        {
            return;
        }
        
        decode_fixed_point(in, out, n);
    }

    // TYPE REMAPPING: internally all compressed representations use integers
    // to distinguish the non-compressed and compressed floating / fixed point numbers
    namespace internal
    {
        template <typename T, std::uint32_t BE, std::uint32_t BM>
        struct fp_remap
        {
            using type = typename FP_NAMESPACE::fp<T>::template format<BE, BM>::type;
        };

        template <>
        struct fp_remap<double, FP_NAMESPACE::fp<double>::default_bits_exponent(), FP_NAMESPACE::fp<double>::default_bits_mantissa()>
        {
            using type = double;
        };

        template <>
        struct fp_remap<float, FP_NAMESPACE::fp<float>::default_bits_exponent(), FP_NAMESPACE::fp<float>::default_bits_mantissa()>
        {
            using type = float;
        };
    }
}

#endif
