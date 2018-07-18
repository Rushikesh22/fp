// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_HPP)
#define FP_HPP

#include <cstdint>
#include <cstring>
#include <cmath>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw 
#endif

namespace FP_NAMESPACE
{
    //! \brief Get maximum
    //!
    //! \tparam T data type
    //! \param in pointer to input sequencemax_bits_compression
    //! \param n length of the input sequence
    //! \return maximum
    template <typename T>
    static inline T scan_max(const T* in, const std::size_t n)
    {
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
    //! \param in pointer to input sequence
    //! \param n length of the input sequence
    //! \return absolute maximum
    template <typename T>
    static inline T scan_absmax(const T* in, const std::size_t n)
    {
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
    //! \param in pointer to input sequence
    //! \param n length of the input sequence
    //! \return minimum
    template <typename T>
    static inline T scan_min(const T* in, const std::size_t n)
    {
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
    //! \param in pointer to input sequence
    //! \param n length of the input sequence
    //! \return absolute minimum
    template <typename T>
    static inline T scan_absmin(const T* in, const std::size_t n)
    {
        T minimum = in[0];

        #pragma omp simd reduction(min : minimum)
        for (std::size_t i = 0; i < n; ++i)
        {
            minimum = std::min(minimum, std::abs(in[i]));
        }

        return minimum;
    }

    //! \brief Class implementing compressed floating point numbers
    //!
    //! \tparam floating point data type to be compressed
    template <typename T>
    class fp
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "error: only 'float' and 'double' type is supported");

        // compressed floating or fixed point numbers have at least 2 bits
        static constexpr std::uint32_t min_bits_compression = 2;

        // compressed floating or fixed point numbers have at most 16 bits
        static constexpr std::uint32_t max_bits_compression = 16;

        //! \brief Class for packing and unpacking of words with 'TB' bits
        //! 
        //! \tparam TB total number of bits (sign + exponent + mantissa)
        template <std::uint32_t TB>
        struct implementation
        {
            static_assert(TB >= min_bits_compression && TB <= max_bits_compression, "error: invalid number of bits for the compressed floating or fixed point representation");

            // 'pack' data type to hold multiple compressed words 
            using type = std::uint64_t;

            // size of the 'pack' data type in bytes
            static constexpr std::size_t pack_bytes = sizeof(type);

            // number of compressed words that fit into the 'pack' data type
            static constexpr std::size_t pack_size = (8 * pack_bytes) / TB;

            //! \brief Number of bytes needed to compress a sequence of 'n' 'TB'-bit words
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of bytes
            static std::size_t memory_footprint_bytes(const std::size_t n)
            {
                const std::size_t num_packs = (n + pack_size - 1) / pack_size;
                return num_packs * pack_bytes;
            }

            //! \brief Number of elements of the 'pack' type needed to compress a sequence of 'n' 'TB'-bit words
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of elements of the 'pack' type
            static std::size_t memory_footprint_elements(const std::size_t n)
            {
                return memory_footprint_bytes(n) / sizeof(type);
            }

            //! \brief Pack compressed 'TB'-bit words
            //!
            //! 'n' x 32-bit words (compressed) are packed into 'n' x 'TB'-bit words that are stored in memory consecutively
            //!
            //! \param out pointer to packed output
            //! \param in pointer to input (only the lowest 'TB' bits of each 32-bit word are used for the packing)
            //! \param n number of 'TB'-bit words to be packed
            static void pack(type* out, const std::uint32_t* in, const std::size_t n)
            {
                // pack as many compressed words as possible in a 'pack'
                
                *out = static_cast<type>(in[0]);
                for (std::size_t i = 1; i < n; ++i)
                {
                    *out |= (static_cast<type>(in[i]) << (i * TB));
                }
            }

            //! \brief Unpack compressed words 'TB'-bit words
            //!
            //! 'n' x 'TB'-bit words are packed into 'n' x 32-bit words (compressed) that are stored in memory consecutively
            //!
            //! \param out pointer to output
            //! \param in pointer to packed input
            //! \param n number of 'TB'-bit words to be unpacked
            static void unpack(std::uint32_t* out, const type* in, const std::size_t n)
            {
                // unpack words into 'out'
                for (std::size_t i = 0; i < n; ++i)
                {
                    out[i] = (*in >> (i * TB)) & get_lower_bits[TB];
                }
            }
        };

        //! \brief Test for exponent and mantissa bits both are larger than 0, and that they add up to at most 15
        //!
        //! \param BE bits exponent
        //! \param BM bits mantissa
        //! \return condition is satisfied or not
        static constexpr bool default_case(const std::uint32_t BE, const std::uint32_t BM)
        {
            return BE > 0 && BM > 0 && (BE + BM) < max_bits_compression;
        }

        //! \brief Test for exponent and mantissa bits match any of the special cases that are defined outside the class
        //!
        //! \param BE bits exponent
        //! \param BM bits mantissa
        //! \return special case or not
        static constexpr bool special_case(const std::uint32_t BE, const std::uint32_t BM)
        {
            return (BE == 8 && BM == 7) || (BE == 0 && (BM == 7 || BM == 11 || BM == 15));
        }

        // bit masks to extract the exponent of a compressed floating point representation with ['BE']['BM'] bits
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

        // bit masks to extract the lowest [n] bits of a word
        static constexpr std::uint32_t get_lower_bits[17] =
                {0x0U, 0x1U, 0x3U, 0x7U, 0xFU, 0x1FU, 0x3FU, 0x7FU, 0xFFU, 0x1FFU, 0x3FFU, 0x7FFU, 0xFFFU, 0x1FFFU, 0x3FFFU, 0x7FFFU, 0xFFFF};

        #if defined(FP_RESCALE)
        // maximum absolute numbers that can be represented with 'BE' bits in the exponent and 16 bits total:
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
        #endif

    public:

        // default number of bits in the exponent (IEEE-754)
        static constexpr std::uint32_t default_bits_exponent();

        // default number of bits in the mantissa (IEEE-754)
        static constexpr std::uint32_t default_bits_mantissa();

        //! \brief Compressed floating point representation with 'BE' exponent and 'BM' mantissa bits
        //!
        //! \tparam BE bits exponent
        //! \tparam BM bits mantissa
        template <std::uint32_t BE, std::uint32_t BM>
        struct format
        {
            static_assert(default_case(BE, BM) || special_case(BE, BM), "error: only floating point compression with 3-16 bits, and fixed point compressione with 8, 12, 16 bits supported");

            // data type for the internal representation of the compressed floating point numbers
            using type = typename implementation<1 + BE + BM>::type;

            //! \brief Number of bytes needed to compress a sequence of 'n' floating point numbers with 'BE' and 'BM' bits + 1 sign bit (implicit)
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of bytes
            static std::size_t memory_footprint_bytes(const std::size_t n)
            {
                #if defined(FP_RESCALE)
                // we need to store the scaling factor as well
                const std::size_t n_scaling_factor = (sizeof(type) * 8 + (1 + BE + BM - 1)) / (1 + BE + BM - 1);
                return implementation<1 + BE + BM>::memory_footprint_bytes(n_scaling_factor + n);
                #else
	            return implementation<1 + BE + BM>::memory_footprint_bytes(n);
                #endif
            }

            //! \brief Number of elements of the 'pack' type needed to compress a sequence of 'n' floating point numbers with 'BE' and 'BM' bits + 1 sign bit (implicit)
            //!
            //! \param n number of floating point numbers to be compressed
            //! \return number of elements of the 'pack' type
            static std::size_t memory_footprint_elements(const std::size_t n)
            {
                return memory_footprint_bytes(n) / sizeof(type);
            }
        };        

        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<!(default_case(BE, BM) && !special_case(BE, BM))>::type>
        static void compress(typename format<BE, BM>::type* out, const T* in, const std::size_t n);
        
        //! \brief Compression (default case)
        //!
        //! The general idea is to truncate both the exponent (after rescaling) and the mantissa, and to pack everything into (1 + 'BE' + 'BM')-bit words
        //! which then are packed for (an almost) contiguous bit stream of compressed floating point numbers
        //!
        //! \tparam BE bits exponent
        //! \tparam BE bits mantissa
        //! \param out pointer to the compressed output bit stream
        //! \param in pointer to the input stream of IEEE-754 floating point numbers
        //! \param n length of the input stream
        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<default_case(BE, BM) && !special_case(BE, BM)>::type>
        static void compress(typename format<BE, BM>::type* out, const T* in, const std::size_t n, const X* ptr = nullptr)
        {
            using fp_type = typename format<BE, BM>::type;

            // bit masks to extract IEEE-754 exponent and mantissa of the single-type (float)
            constexpr std::uint32_t get_exponent = 0x7F800000U;
            constexpr std::uint32_t get_mantissa = 0x007FFFFFU;

            // minimum and maximum number of the exponent with 'BE' bits
            constexpr std::uint32_t range_min = 127 - ((0x1 << (BE - 1)) - 1);
            constexpr std::uint32_t range_max = 127 + (0x1 << (BE - 1));

            #if defined(FP_RESCALE)
            // for rescaling, first determine the absolute maximum value among all uncompressed floating point numbers...
            const float abs_max = static_cast<float>(scan_absmax(in, n));
            // calculate the rescaling factor...
            const float a = 0.98F * scaling_factor[BE] / abs_max;
            // and place it as the 1st element to the output stream
            float* fptr_out = reinterpret_cast<float*>(out);
            fptr_out[0] = 1.0F / a;
            // all compressed floating point numbers are placed after the rescaling factor
            fp_type* ptr_out = reinterpret_cast<fp_type*>(&fptr_out[1]);
            #else
            fp_type* ptr_out = out;
            #endif

            constexpr std::size_t pack_bytes = implementation<1 + BE + BM>::pack_bytes;
            constexpr std::size_t pack_size = implementation<1 + BE + BM>::pack_size;

            // in case of T = 'double', there is an implicit down cast to 'float', that is,
            // all computation below is on 32-bit words!
            std::uint32_t buffer[pack_size];
            float* fptr_buffer = reinterpret_cast<float*>(&buffer[0]);

            // process all input data in chunks of size 'pack_size'
            for (std::size_t i = 0; i < n; i += pack_size, ptr_out += (pack_bytes / sizeof(fp_type)))
            {
                // number of elements to process
                const std::size_t ii_max = std::min(n - i, pack_size);

                // load the floating point numbers into the local buffer
                #if defined(FP_RESCALE)
                // and apply the rescaling
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    fptr_buffer[ii] = static_cast<float>(in[i + ii]) * a;
                }
                #else
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    fptr_buffer[ii] = static_cast<float>(in[i + ii]);
                }
                #endif

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

                // pack the compressed floating point numbers in 'buffer'
                implementation<1 + BE + BM>::pack(ptr_out, buffer, ii_max);
            }
        }

        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<!(default_case(BE, BM) && !special_case(BE, BM))>::type>
        static void decompress(T* out, const typename format<BE, BM>::type* in, const std::size_t n);

        //! \brief Decompression (default case)
        //!
        //! \tparam BE bits exponent
        //! \tparam BE bits mantissa
        //! \param out pointer to the decompressed output stream of IEEE-754 floating point numbers
        //! \param in pointer to the compressed input bit stream
        //! \param n length of the input stream
        template <std::uint32_t BE, std::uint32_t BM, typename X = typename std::enable_if<default_case(BE, BM) && !special_case(BE, BM)>::type>
        static void decompress(T* out, const typename format<BE, BM>::type* in, const std::size_t n, const X* ptr = nullptr)
        {
            using fp_type = typename format<BE, BM>::type;

            #if defined(FP_RESCALE)
            // recover the scaling factor (1st element) from the input stream
            const float* fptr_in = reinterpret_cast<const float*>(in);
            const float a = fptr_in[0];
            // and move on to the compressed and packed floating point values
            const fp_type* ptr_in = reinterpret_cast<const fp_type*>(&fptr_in[1]); 
            #else
            const fp_type* ptr_in = in;
            #endif

            constexpr std::size_t pack_bytes = implementation<1 + BE + BM>::pack_bytes;
            constexpr std::size_t pack_size = implementation<1 + BE + BM>::pack_size;

            // in case of T = 'double', there is an implicit up cast from 'float' to 'double', that is,
            // all computation below is on 32-bit words!
            std::uint32_t buffer[pack_size];
            const float* fptr_buffer = reinterpret_cast<const float*>(&buffer[0]);

            for (std::size_t i = 0; i < n; i += pack_size, ptr_in += (pack_bytes / sizeof(fp_type)))
            {
                // number of elements to process
                const std::size_t ii_max = std::min(n - i, pack_size);

                // unpack the compressed floating point numbers into 'buffer'
                implementation<1 + BE + BM>::unpack(buffer, ptr_in, ii_max);

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

                // store the floating point numbers
                #if defined(FP_RESCALE)
                // and apply the rescaling
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    out[i + ii] = static_cast<T>(fptr_buffer[ii] * a);
                }
                #else
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    out[i + ii] = static_cast<T>(fptr_buffer[ii]);
                }
                #endif
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
        using type = double;

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
    inline void fp<double>::compress<11, 52>(double* out, const double* in, const std::size_t n)
    {
        if (in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = in[i];
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<11, 52>(double* out, const double* in, const std::size_t n)
    {
        if (in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = in[i];
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // NO COMPRESSION: IEEE-754 single type
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::format<8, 23>
    {
        using type = float;

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
    inline void fp<float>::compress<8, 23>(float* out, const float* in, const std::size_t n)
    {
        if (in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = in[i];
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<8, 23>(float* out, const float* in, const std::size_t n)
    {
        if (in == out)
        {
            return;
        }

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = in[i];
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 32 bits: 8 bit exponent and 23 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<8, 23>
    {
        using type = float;

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
    inline void fp<double>::compress<8, 23>(float* out, const double* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<float>(in[i]);
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 23>(double* out, const float* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<double>(in[i]);
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
    inline void fp<double>::compress<8, 7>(typename format<8, 7>::type* out, const double* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const float f_tmp = in[i];
            const std::uint32_t i_tmp = *reinterpret_cast<const std::uint32_t*>(&f_tmp) >> 16;
            out[i] = i_tmp;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 7>(double* out, const typename format<8, 7>::type* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint32_t tmp = static_cast<std::uint32_t>(in[i]) << 16;
            out[i] = static_cast<double>(*reinterpret_cast<const float*>(&tmp));
        }
    }
    
    template <>
    template <>
    struct fp<float>::format<8, 7>
    {
        using type = std::uint16_t;

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
    inline void fp<float>::compress<8, 7>(typename format<8, 7>::type* out, const float* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint32_t tmp = *reinterpret_cast<const std::uint32_t*>(&in[i]) >> 16;
            out[i] = tmp;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<8, 7>(float* out, const typename format<8, 7>::type* in, const std::size_t n)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint32_t tmp = static_cast<std::uint32_t>(in[i]) << 16;
            out[i] = *reinterpret_cast<const float*>(&tmp);
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

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 15>(typename format<0, 15>::type* out, const double* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 15>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (static_cast<float>(in[i]) - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 15>(double* out, const typename format<0, 15>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 15>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        }
    }

    template <>
    template <>
    struct fp<float>::format<0, 15>
    {
        using type = std::uint16_t;

        static constexpr std::size_t pack_bytes = 2 * 32;

        static constexpr std::size_t pack_size = 1 * 16;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 15>(typename format<0, 15>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 15>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (in[i] - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 15>(float* out, const typename format<0, 15>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 15>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i] * b + a;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 12 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 11>
    {
        using type = std::uint8_t;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + ((n + 1) / 2) * 3 * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 11>(typename format<0, 11>::type* out, const double* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 11>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        const std::size_t i_max = (n / 2) * 2;
        for (std::size_t i = 0; i < i_max; i += 2)
        {
            const std::uint32_t tmp = static_cast<std::uint32_t>((static_cast<float>(in[i + 0]) - a) * b) | (static_cast<std::uint32_t>((static_cast<float>(in[i + 1]) - a) * b) << 12);
            *reinterpret_cast<std::uint32_t*>(ptr_out) = tmp;
            ptr_out += 3;
        }

        if (n % 2)
        {
            const std::uint16_t tmp = static_cast<std::uint16_t>((in[n - 1] - a) * b);
            *reinterpret_cast<std::uint16_t*>(ptr_out) = tmp;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 11>(double* out, const typename format<0, 11>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 11>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        const std::size_t i_max = (n / 2) * 2;
        for (std::size_t i = 0, k = 0; i < i_max; i += 2, k += 3)
        {
            const std::uint32_t tmp = *(reinterpret_cast<const std::uint32_t*>(&ptr_in[k]));
            out[i + 0] = (tmp & 0xFFFU) * b + a;
            out[i + 1] = ((tmp >> 12) & 0xFFFU) * b + a;
            ptr_in += 3;
        }

        if (n % 2)
        {
            const std::uint16_t tmp = *(reinterpret_cast<const std::uint16_t*>(ptr_in));
            out[n - 1] = (tmp & 0xFFFU) * b + a;
        }
    }
    
    template <>
    template <>
    struct fp<float>::format<0, 11>
    {
        using type = std::uint8_t;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + ((n + 1) / 2) * 3 * sizeof(type));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 11>(typename format<0, 11>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 11>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        const std::size_t i_max = (n / 2) * 2;
        for (std::size_t i = 0; i < i_max; i += 2)
        {
            const std::uint32_t tmp = static_cast<std::uint32_t>((in[i + 0] - a) * b) | (static_cast<std::uint32_t>((in[i + 1] - a) * b) << 12);
            *(reinterpret_cast<std::uint32_t*>(ptr_out)) = tmp;
            ptr_out += 3;
        }

        if (n % 2)
        {
            const std::uint16_t tmp = static_cast<std::uint16_t>((in[n - 1] - a) * b);
            *(reinterpret_cast<std::uint16_t*>(ptr_out)) = tmp;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 11>(float* out, const typename format<0, 11>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 11>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        const std::size_t i_max = (n / 2) * 2;
        for (std::size_t i = 0; i < i_max; i += 2)
        {
            const std::uint32_t tmp = *(reinterpret_cast<const std::uint32_t*>(ptr_in));
            out[i + 0] = (tmp & 0xFFFU) * b + a;
            out[i + 1] = ((tmp >> 12) & 0xFFFU) * b + a;
            ptr_in += 3;
        }

        if (n % 2)
        {
            const std::uint16_t tmp = *(reinterpret_cast<const std::uint16_t*>(ptr_in));
            out[n - 1] = (tmp & 0xFFFU) * b + a;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 8 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 7>
    {
        using type = std::uint8_t;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 7>(typename format<0, 7>::type* out, const double* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 7>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (static_cast<float>(in[i]) - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 7>(double* out, const typename format<0, 7>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 7>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        }
    }

    template <>
    template <>
    struct fp<float>::format<0, 7>
    {
        using type = std::uint8_t;

        static inline std::size_t memory_footprint_bytes(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }

        static inline std::size_t memory_footprint_elements(const std::size_t n)
        {
            return memory_footprint_bytes(n) / sizeof(type);
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 7>(typename format<0, 7>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 7>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (in[i] - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 7>(float* out, const typename format<0, 7>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 7>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i)
        {
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        } 
    }
}

#endif
