// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_HPP)
#define FP_HPP

#include <cstdint>
#include <cstring>

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw 
#endif

namespace FP_NAMESPACE
{
    template <typename T>
    static inline T scan_max(const T* in, const std::size_t n)
    {
        T maximum = in[0];
        for (std::size_t i = 1; i < n; ++i)
        {
            maximum = (maximum < in[i] ? in[i] : maximum);
        }
        return maximum;
    }

    template <typename T>
    static inline T scan_absmax(const T* in, const std::size_t n)
    {
        T maximum = in[0];
        for (std::size_t i = 1; i < n; ++i)
        {
   	    maximum = (maximum < std::abs(in[i]) ? std::abs(in[i]) : maximum);
        }
        return maximum;
    }

    template <typename T>
    static inline T scan_min(const T* in, const std::size_t n)
    {
        T minimum = in[0];
        for (std::size_t i = 1; i < n; ++i)
        {
            minimum = (minimum > in[i] ? in[i] : minimum);
        }
        return minimum;
    }

    template <typename T>
    class fp
    {
        template <std::int32_t TB>
        struct implementation
        {
            using type = std::int64_t;

            static constexpr std::size_t chunk_bytes = sizeof(type);

            static constexpr std::size_t chunk_size = (8 * chunk_bytes) / TB;

            static std::size_t get_memory_footprint(const std::size_t n)
            {
                return ((n + chunk_size - 1) / chunk_size) * chunk_bytes;
            }

            static void compress_and_store(type* out, std::int32_t* in, const std::size_t n)
            {
                // accumulate as many compressed words as possible in a word of type 'type'
                type in_compressed = in[0];
                for (std::size_t i = 1; i < n; ++i)
                {
                    in_compressed |= (static_cast<type>(in[i]) << (i * TB));
                }

                // store data
                std::memcpy(&out[0], &in_compressed, chunk_bytes);
            }

            static void load_and_decompress(std::int32_t* out, const type* in, const std::size_t n)
            {
                // load compressed data
                type in_compressed;
                std::memcpy(&in_compressed, &in[0], chunk_bytes);

                // decompress the content into x[]
                for (std::size_t i = 0; i < n; ++i)
                {
                    out[i] = (in_compressed >> (i * TB)) & get_lower_bits[TB];
                }
            }
        };

        static constexpr std::int32_t get_exponent[17][17] = {
                {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x0},
                {0x0003, 0x0006, 0x000C, 0x0018, 0x0030, 0x0060, 0x00C0, 0x0180, 0x0300, 0x0600, 0x0C00, 0x1800, 0x3000, 0x6000, 0x0, 0x0},
                {0x0007, 0x000E, 0x001C, 0x0038, 0x0070, 0x00E0, 0x01C0, 0x0380, 0x0700, 0x0E00, 0x1C00, 0x3800, 0x7000, 0x0, 0x0, 0x0},
                {0x000F, 0x001E, 0x003C, 0x0078, 0x00F0, 0x01E0, 0x03C0, 0x0780, 0x0F00, 0x1E00, 0x3C00, 0x7800, 0x0, 0x0, 0x0, 0x0},
                {0x001F, 0x003E, 0x007C, 0x00F8, 0x01F0, 0x03E0, 0x07C0, 0x0F80, 0x1F00, 0x3E00, 0x7C00, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x003F, 0x007E, 0x00FC, 0x01F8, 0x03F0, 0x07E0, 0x0FC0, 0x1F80, 0x3F00, 0x7E00, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x007F, 0x00FE, 0x01FC, 0x03F8, 0x07F0, 0x0FE0, 0x1FC0, 0x3F80, 0x7F00, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x00FF, 0x01FE, 0x03FC, 0x07F8, 0x0FF0, 0x1FE0, 0x3FC0, 0x7F80, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x01FF, 0x03FE, 0x07FC, 0x0FF8, 0x1FF0, 0x3FE0, 0x7FC0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x03FF, 0x07FE, 0x0FFC, 0x1FF8, 0x3FF0, 0x7FE0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x07FF, 0x0FFE, 0x1FFC, 0x3FF8, 0x7FF0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x0FFF, 0x1FFE, 0x3FFC, 0x7FF8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x1FFF, 0x3FFE, 0x7FFC, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x3FFF, 0x7FFE, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
                {0x7FFF, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}};

        static constexpr std::int32_t get_lower_bits[17] =
                {0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};

        #if defined(FP_RESCALE)
        // for be = 1..8 : value = (1 - 2 ^ (15 - be)) * 2 ^ (2 ^ (be - 1))
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

        static constexpr std::int32_t default_bits_exponent();

        static constexpr std::int32_t default_bits_mantissa();
        
        template <std::int32_t BE, std::int32_t BM>
        struct format
        {
            using type = typename implementation<1 + BE + BM>::type;

            static std::size_t get_memory_footprint(const std::size_t n)
            {
                #if defined(FP_RESCALE)
                return (sizeof(float) + implementation<1 + BE + BM>::get_memory_footprint(n));
                #else
	            return implementation<1 + BE + BM>::get_memory_footprint(n);
                #endif
            }
        };

        template <std::int32_t BE, std::int32_t BM>
        static void compress(typename format<BE, BM>::type* out, const T* in, const std::size_t n)
        {
            using fp_t = typename format<BE, BM>::type;

            constexpr std::int32_t get_exponent = 0x7F800000;
            constexpr std::int32_t get_mantissa = 0x007FFFFF;

            constexpr std::int32_t range_max = 127 + (0x1 << (BE - 1));
            constexpr std::int32_t range_min = 127 - ((0x1 << (BE - 1)) - 1);

            constexpr std::size_t chunk_bytes = implementation<1 + BE + BM>::chunk_bytes;
            constexpr std::size_t chunk_size = implementation<1 + BE + BM>::chunk_size;
            std::int32_t buffer[chunk_size];

            float* fptr_buffer = reinterpret_cast<float*>(&buffer[0]);
            #if defined(FP_RESCALE)        
            const float abs_max = static_cast<float>(scan_absmax(in, n));
            const float a = (0.99F * scaling_factor[BE]) / abs_max;
            float* fptr_out = reinterpret_cast<float*>(out);
            fptr_out[0] = a;
            fp_t* ptr_out = reinterpret_cast<fp_t*>(&fptr_out[1]);
            #else
            fp_t* ptr_out = out; 
            #endif

            for (std::size_t i = 0, k = 0; i < n; i += chunk_size, k += (chunk_bytes / sizeof(fp_t)))
            {
                // how many elements to process?
                const std::size_t ii_max = std::min(n - i, chunk_size);

                // load data
                #if defined(FP_RESCALE)
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

                // compress all 32 bit words individually: resulting bit pattern begins at bit 0
                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    const std::int32_t current_element = buffer[ii];
                    const std::int32_t exponent = (current_element & get_exponent) >> 23;
                    #if defined(FP_RESCALE)
                    const std::int32_t sat_exponent = std::max(exponent, range_min);
                    #else
                    const std::int32_t sat_exponent = std::max(std::min(exponent, range_max), range_min);
                    #endif
                    const std::int32_t new_exponent = (sat_exponent - range_min) << BM;
                    const std::int32_t new_mantissa = (current_element & get_mantissa) >> (23 - BM);
                    const std::int32_t new_sign = (current_element & 0x80000000) >> (31 - (BE + BM));

                    buffer[ii] = (new_sign | new_exponent | new_mantissa);
                }

                // compress and store the data
                implementation<1 + BE + BM>::compress_and_store(&ptr_out[k], &buffer[0], ii_max);
            }
        }

        template <std::int32_t BE, std::int32_t BM>
        static void decompress(T* out, const typename format<BE, BM>::type* in, const std::size_t n)
        {
            using fp_t = typename format<BE, BM>::type;

            constexpr std::size_t chunk_bytes = implementation<1 + BE + BM>::chunk_bytes;
            constexpr std::size_t chunk_size = implementation<1 + BE + BM>::chunk_size;
            std::int32_t buffer[chunk_size];

            float* fptr_buffer = reinterpret_cast<float*>(&buffer[0]);
            #if defined(FP_RESCALE)
            const float* fptr_in = reinterpret_cast<const float*>(in);
            const float a = 1.0F / fptr_in[0];
            const fp_t* ptr_in = reinterpret_cast<const fp_t*>(&fptr_in[1]); 
            #else
            const fp_t* ptr_in = in;
            #endif

            for (std::size_t i = 0, k = 0; i < n; i += chunk_size, k += (chunk_bytes / sizeof(fp_t)))
            {
                // how many elements to process?
                const std::size_t ii_max = std::min(n - i, chunk_size);

                // load and decompress the data
                implementation<1 + BE + BM>::load_and_decompress(&buffer[0], &ptr_in[k], ii_max);

                for (std::size_t ii = 0; ii < ii_max; ++ii)
                {
                    const std::int32_t current_element = buffer[ii];
                    const std::int32_t exponent = (current_element & get_exponent[BE][BM]) >> BM;
                    const std::int32_t mantissa = (current_element & get_lower_bits[BM]);
                    const std::int32_t new_mantissa = mantissa << (31 - (8 + BM));
                    const std::int32_t new_exponent = (exponent - ((0x1 << (BE - 1)) - 1) + 127) << (31 - 8);
                    const std::int32_t new_sign = (buffer[ii] << (31 - (BE + BM))) & 0x80000000;

                    buffer[ii] = (new_sign | new_exponent | new_mantissa);
                }

                // store data
                #if defined(FP_RESCALE)
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
    constexpr std::int32_t fp<double>::default_bits_exponent()
    {
        return 11;
    }

    template <>
    constexpr std::int32_t fp<double>::default_bits_mantissa()
    {
        return 52;
    }

    template <>
    constexpr std::int32_t fp<float>::default_bits_exponent()
    {
        return 8;
    }

    template <>
    constexpr std::int32_t fp<float>::default_bits_mantissa()
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

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
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

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
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
        using type = std::int16_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<8, 23>(float* out, const double* in, const std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<float>(in[i]);
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 23>(double* out, const float* in, const std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<double>(in[i]);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 16 bits: 11 bit exponent and 4 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<11, 4>
    {
        using type = std::int16_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<11, 4>(typename format<11, 4>::type* out, const double* in, const std::size_t n)
    {
        using in_t = typename format<11, 4>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(in);

        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[4 * i + 3];
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<11, 4>(double* out, const typename format<11, 4>::type* in, const std::size_t n)
    {
        using out_t = typename format<11, 4>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(out);

        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = 0.0;
            ptr_out[4 * i + 3] = in[i];
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 16 bits: 8 bit exponent and 7 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::format<8, 7>
    {
        using type = std::int16_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<8, 7>(typename format<8, 7>::type* out, const float* in, const std::size_t n)
    {
        using in_t = typename format<8, 7>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(in);

        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[2 * i + 1];
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<8, 7>(float* out, const typename format<8, 7>::type* in, const std::size_t n)
    {
        using out_t = typename format<8, 7>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(out);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[2 * i + 0] = 0;
            ptr_out[2 * i + 1] = in[i];
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 16 bits: 8 bit exponent and 7 bit mantissa
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<8, 7>
    {
        using type = std::int16_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<8, 7>(typename format<8, 7>::type* out, const double* in, const std::size_t n)
    {
        constexpr std::size_t len = 64;
        float tmp[len];

        using in_t = typename format<8, 7>::type;
        const in_t* ptr_tmp = reinterpret_cast<const in_t*>(&tmp[0]);

        for (std::size_t i = 0; i < n; i += len)
        {
            const std::size_t ii_max = std::min(len, n - i);

            for (std::size_t ii = 0; ii < ii_max; ++ii)
            {
                tmp[ii] = in[i + ii];
            }

            for (std::size_t ii = 0; ii < ii_max; ++ii)
            {
                out[i + ii] = ptr_tmp[2 * ii + 1];
            }
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<8, 7>(double* out, const typename format<8, 7>::type* in, const std::size_t n)
    {
        constexpr std::size_t len = 64;
        float tmp[len];

        using out_t = typename format<8, 7>::type;
        out_t* ptr_tmp = reinterpret_cast<out_t*>(&tmp[0]);

        for (std::size_t i = 0; i < n; i += len)
        {
            const std::size_t ii_max = std::min(len, n - i);

            for (std::size_t ii = 0; ii < ii_max; ++ii)
            {
                ptr_tmp[2 * ii + 0] = 0;
                ptr_tmp[2 * ii + 1] = in[i + ii];
            }
            
            for (std::size_t ii = 0; ii < ii_max; ++ii)
            {
                out[i + ii] = tmp[ii];
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 12 bits
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::implementation<12>
    {
        using type = std::int8_t;

        static constexpr std::size_t chunk_bytes = 3 * 32;

        static constexpr std::size_t chunk_size = 2 * 32;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return ((n + 1) / 2) * 3;
        }

        static inline void compress_and_store(type* out, std::int32_t* in, const std::size_t n)
        {
            // chain compressed words: 2 x 12 bit -> 24 bit -> 3 x 8 bit
            type* in_compressed = reinterpret_cast<type*>(&in[0]);
            for (std::size_t i = 0, k = 0; i < n; i += 2, k += 3)
            {
                const std::int32_t tmp = in[i] | (in[i + 1] << 12);
                std::int32_t* in_current = reinterpret_cast<std::int32_t*>(&in_compressed[k]);
                *in_current = tmp;
            }

            // store data
            std::memcpy(&out[0], &in[0], ((n + 1) / 2) * 3);
        }

        static inline void load_and_decompress(std::int32_t* out, const type* in, const std::size_t n)
        {
            // load compressed data and decompress into x[]
            for (std::size_t i = 0, k = 0; i < n; i += 2, k += 3)
            {
                const std::int32_t* in_compressed = reinterpret_cast<const std::int32_t*>(&in[k]);
                const std::int32_t tmp = *in_compressed;
                out[i] = (tmp & 0x00000FFF);
                out[i + 1] = ((tmp >> 12) & 0x00000FFF);
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 10 bits
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::implementation<10>
    {
        using type = std::int8_t;

        static constexpr std::size_t chunk_bytes = 5 * 32;

        static constexpr std::size_t chunk_size = 4 * 32;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return ((n + 3) / 4) * 5;
        }

        static inline void compress_and_store(type* out, std::int32_t* in, const std::size_t n)
        {
            // chain compressed words: 4 x 10 bit -> 40 bit -> 5 x 8 bit
            type* in_compressed = reinterpret_cast<type*>(&in[0]);
            for (std::size_t i = 0, k = 0; i < n; i += 4, k += 5)
            {
                const std::int32_t tmp = in[i] | (in[i + 1] << 10) | (in[i + 2] << 20);
                std::int64_t* in_current = reinterpret_cast<std::int64_t*>(&in_compressed[k]);
                *in_current = (static_cast<std::int64_t>(tmp) | (static_cast<std::int64_t>(in[i + 3]) << 30));
            }

            // store data
            std::memcpy(&out[0], &in[0], ((n + 3) / 4) * 5);
        }

        static inline void load_and_decompress(std::int32_t* out, const type* in, const std::size_t n)
        {
            // load compressed data and decompress into x[]
            for (std::size_t i = 0, k = 0; i < n; i += 4, k += 5)
            {
                const std::int64_t* in_compressed = reinterpret_cast<const std::int64_t*>(&in[k]);
                const std::int64_t tmp64 = *in_compressed;
                const std::int32_t tmp32 = static_cast<std::int32_t>(tmp64);
                out[i] = (tmp32 & 0x000003FF);
                out[i + 1] = ((tmp32 >> 10) & 0x000003FF);
                out[i + 2] = ((tmp32 >> 20) & 0x000003FF);
                out[i + 3] = static_cast<std::int32_t>((tmp64 >> 30) & 0x00000000000003FF);
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATION for 8 bits
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::implementation<8>
    {
        using type = std::int8_t;

        static constexpr std::size_t chunk_bytes = 4 * 32;

        static constexpr std::size_t chunk_size = 4 * 32;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return n;
        }

        static inline void compress_and_store(type* out, std::int32_t* in, const std::size_t n)
        {
            // chain compressed words
            type* ptr_in = reinterpret_cast<type*>(&in[0]);
            for (std::size_t i = 0; i < n; ++i)
            {
                out[i] = ptr_in[4 * i];
            }
        }

        static inline void load_and_decompress(std::int32_t* out, const type* in, const std::size_t n)
        {
            // load compressed data and decompress into x[]
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::int32_t tmp = in[i];
                out[i] = tmp;
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<float>::format<0, 16>
    {
        using type = std::uint16_t;

        static constexpr std::size_t chunk_bytes = 2 * 32;

        static constexpr std::size_t chunk_size = 1 * 16;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 16>(typename format<0, 16>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 16>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (in[i] - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 16>(float* out, const typename format<0, 16>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 16>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i] * b + a;
        }
    }

    template <>
    template <>
    struct fp<double>::format<0, 16>
    {
        using type = std::uint16_t;

        static constexpr std::size_t chunk_bytes = 2 * 32;

        static constexpr std::size_t chunk_size = 1 * 16;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 16>(typename format<0, 16>::type* out, const double* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 16>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (static_cast<float>(in[i]) - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 16>(double* out, const typename format<0, 16>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 16>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<double>(ptr_in[i] * b + a);
        }
    }

    template <>
    template <>
    struct fp<float>::format<0, 12>
    {
        using type = std::uint8_t;

        static constexpr std::size_t chunk_bytes = 3 * 32;

        static constexpr std::size_t chunk_size = 2 * 16;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + ((n + 1) / 2) * 3 * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 12>(typename format<0, 12>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 12>::type;
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
    inline void fp<float>::decompress<0, 12>(float* out, const typename format<0, 12>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 12>::type;
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

    template <>
    template <>
    struct fp<float>::format<0, 8>
    {
        using type = std::uint8_t;

        static constexpr std::size_t chunk_bytes = 1 * 32;

        static constexpr std::size_t chunk_size = 1 * 16;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<0, 8>(typename format<0, 8>::type* out, const float* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 8>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (in[i] - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<float>::decompress<0, 8>(float* out, const typename format<0, 8>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 8>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = ptr_in[i] * b + a;
        } 
        /*
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 8>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        constexpr std::size_t bs = 32;
        std::int32_t buffer[bs];
        for (std::size_t ii = 0; ii < bs; ++ii)
        {
            buffer[ii] = 0;
        }
        std::uint8_t* ptr_buffer = reinterpret_cast<std::uint8_t*>(&buffer[0]);

        const size_t i_max = (n / bs) * bs;
        for (std::size_t i = 0; i < i_max; i += bs)
        {
            for (std::size_t ii = 0; ii < bs; ++ii)
            {
                ptr_buffer[4 * ii] = ptr_in[i + ii];
            }
            for (std::size_t ii = 0; ii < bs; ++ii)
            {
                out[i + ii] = buffer[ii] * b + a;
            }
        }
        for (std::size_t i = i_max; i < n; ++i)
        { 
            out[i] = ptr_in[i] * b + a;
        }
        */
    }

    template <>
    template <>
    struct fp<double>::format<0, 8>
    {
        using type = std::uint8_t;

        static constexpr std::size_t chunk_bytes = 1 * 32;

        static constexpr std::size_t chunk_size = 1 * 16;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + (n * sizeof(type)));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 8>(typename format<0, 8>::type* out, const double* in, const std::size_t n)
    {
        float minimum = scan_min(in, n);
        float maximum = scan_max(in, n);

        const float a = minimum;
        const float b = 0xFFU / (maximum - a);

        float* fptr_out = reinterpret_cast<float*>(out);
        fptr_out[0] = a;
        fptr_out[1] = 1.0F / b;

        using out_t = typename format<0, 8>::type;
        out_t* ptr_out = reinterpret_cast<out_t*>(&fptr_out[2]);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr_out[i] = (static_cast<float>(in[i]) - a) * b;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 8>(double* out, const typename format<0, 8>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 8>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<double>(ptr_in[i] * b + a);
        }
    }
}

#endif
