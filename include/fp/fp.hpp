// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FP_HPP)
#define FP_HPP

#include <cstdint>
#include <cstring>
#include <immintrin.h>

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
        using type = float;

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
        using type = std::uint16_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<11, 4>(typename format<11, 4>::type* out, const double* in, const std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint64_t tmp = *reinterpret_cast<const std::uint64_t*>(&in[i]) >> 48;
            out[i] = tmp;
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<11, 4>(double* out, const typename format<11, 4>::type* in, const std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint64_t tmp = static_cast<std::uint64_t>(in[i]) << 48;
            out[i] = *reinterpret_cast<const double*>(&tmp);
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

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<8, 7>(typename format<8, 7>::type* out, const double* in, const std::size_t n)
    {
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

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (n * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<float>::compress<8, 7>(typename format<8, 7>::type* out, const float* in, const std::size_t n)
    {
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
    struct fp<double>::format<0, 16>
    {
        using type = std::uint16_t;

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
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        }
    }

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

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 12 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 12>
    {
        using type = std::uint8_t;

        static inline std::size_t get_memory_footprint(const std::size_t n)
        {
            return (2 * sizeof(float) + ((n + 1) / 2) * 3 * sizeof(type));
        }
    };

    template <>
    template <>
    inline void fp<double>::compress<0, 12>(typename format<0, 12>::type* out, const double* in, const std::size_t n)
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

        for (std::size_t i = 0, k = 0; i < i_max; i += 2, k += 3)
        {
            const std::uint32_t tmp = static_cast<std::uint32_t>((static_cast<float>(in[i + 0]) - a) * b) | (static_cast<std::uint32_t>((static_cast<float>(in[i + 1]) - a) * b) << 12);
            *reinterpret_cast<std::uint32_t*>(&ptr_out[k]) = tmp;
        }

        if (n % 2)
        {
            ptr_out[(n / 2) * 3] = static_cast<std::uint8_t>((in[n - 1] - a) * b);
        }
    }

    template <>
    template <>
    inline void fp<double>::decompress<0, 12>(double* out, const typename format<0, 12>::type* in, const std::size_t n)
    {
        const float* fptr_in = reinterpret_cast<const float*>(in);
        const float a = fptr_in[0];
        const float b = fptr_in[1];

        using in_t = typename format<0, 12>::type;
        const in_t* ptr_in = reinterpret_cast<const in_t*>(&fptr_in[2]);
        
        const std::size_t i_max = (n / 2) * 2;
        for (std::size_t i = 0, k = 0; i < i_max; i += 2, k += 3)
        {
            const std::uint32_t i_tmp = *(reinterpret_cast<const std::uint32_t*>(&ptr_in[k]));
            const float f_tmp_1 = i_tmp & 0xFFF;
            const float f_tmp_2 = (i_tmp >> 12) & 0xFFF;
            out[i + 0] = f_tmp_1 * b + a;
            out[i + 1] = f_tmp_2 * b + a;
        }

        if (n % 2)
        {
            const float f_tmp = static_cast<float>(*(reinterpret_cast<const std::uint8_t*>(&ptr_in[(n / 2) * 3])));
            out[n - 1] = f_tmp * b + a;
        }
    }
    
    template <>
    template <>
    struct fp<float>::format<0, 12>
    {
        using type = std::uint8_t;

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

    ////////////////////////////////////////////////////////////////////////////////////
    // SPECIALIZATIONS: fixed precision 8 bit
    ////////////////////////////////////////////////////////////////////////////////////
    template <>
    template <>
    struct fp<double>::format<0, 8>
    {
        using type = std::uint8_t;

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
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        }
    }

    template <>
    template <>
    struct fp<float>::format<0, 8>
    {
        using type = std::uint8_t;

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
            const float tmp = ptr_in[i];
            out[i] = tmp * b + a;
        } 
    }
}

#endif
