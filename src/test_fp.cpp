// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <vector>
#include <fp/fp.hpp>
#include <misc/misc.hpp>

//using real_t = float;
using real_t = double;
constexpr double max_dev = 1.0E-3;

constexpr bool implementation_available(const std::size_t bm, const std::size_t be)
{
    const bool between_2bit_and_16bit = (be > 0 && bm > 0 && (be + bm) < 16);
    const bool no_compression = (be == fw::ieee754_fp<real_t>::be && bm == fw::ieee754_fp<real_t>::bm);
    const bool double_to_float = (std::is_same<real_t, double>::value && be == fw::ieee754_fp<float>::be && bm == fw::ieee754_fp<float>::bm);
    const bool fixed_point = (be == 0 && (bm == 8 || bm == 16));

    return between_2bit_and_16bit || no_compression || double_to_float || fixed_point;
}

template <std::size_t bm, std::size_t be, typename X = typename std::enable_if<!implementation_available(bm, be)>::type>
bool test_compression(const std::size_t n, const std::size_t seed = 1)
{
    std::cout << "not implementation_available" << std::endl;
    return false;
}

template <std::size_t bm, std::size_t be, typename X = typename std::enable_if<implementation_available(bm, be)>::type>
void test_compression(const std::size_t n, const std::size_t seed, const X* ptr = nullptr)
{
    using fp_format = fw::fp_stream<bm, be>;
    using fp_type = typename fp_format::type;
    const std::size_t n_compressed = fp_format::memory_footprint_elements(n);

    std::vector<real_t> data(n);
    std::vector<fp_type> data_compressed(n_compressed);
    std::vector<real_t> data_decompressed(n);

    srand48(seed);
    for (std::size_t i = 0; i < n; ++i)
    {
        data[i] = 2.0 * drand48() - 1.0;
    }

    fp_format::compress(&data[0], &data_compressed[0], n);
    fp_format::decompress(&data_compressed[0], &data_decompressed[0], n);

    real_t a = data[0];
    real_t b = data_decompressed[0];
    double dev = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double tmp = std::abs((data_decompressed[i] - data[i]) / data[i]);
        if (tmp > dev)
        {
            dev = tmp;
            a = data[i];
            b = data_decompressed[i];
        }
    }

    if (dev > max_dev)
    {
        std::cout << "failed\t[" << a << " vs. " << b << "]" << std::endl;
    }
    else
    {
        std::cout << "passed" << std::endl;
    }
}

int main(int argc, char** argv)
{
    const std::size_t bits_total = (argc > 1 ? atoi(argv[1]) : 16);
    const std::size_t n = (argc > 2 ? atoi(argv[2]) : 128);
    const std::size_t seed = (argc > 3 ? atoi(argv[3]) : 1);
    
    fw::loop<16, 16>::body([&bits_total, &n, &seed](auto& x_1, auto& x_2)
    { 
        constexpr std::size_t be = x_1.value;
        constexpr std::size_t bm = x_2.value;
        constexpr std::size_t be_max = fw::ieee754_fp<float>::be;
        if (implementation_available(bm, be) && (be <= be_max) && (bm > 1) && (be + bm) == (bits_total - 1))
        {
            std::cout << "bits(1," << be << "," << bm << ")\t";
            test_compression<bm, be>(n, seed);
        }
    });

    std::cout << "bits(1," << fw::ieee754_fp<real_t>::be << "," << fw::ieee754_fp<real_t>::bm << ")\t";
    test_compression<fw::ieee754_fp<real_t>::bm, fw::ieee754_fp<real_t>::be>(n, seed);

    if (std::is_same<real_t, double>::value)
    {
        std::cout << "bits(1," << fw::ieee754_fp<float>::be << "," << fw::ieee754_fp<float>::bm << ")\t";
        test_compression<fw::ieee754_fp<float>::bm, fw::ieee754_fp<float>::be>(n, seed);
    }

    return 0;
}