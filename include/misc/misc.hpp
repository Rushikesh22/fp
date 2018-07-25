// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(MISC_HPP)
#define MISC_HPP

#if !defined(FP_NAMESPACE)
#define FP_NAMESPACE fw 
#endif

namespace FP_NAMESPACE
{
    /////////////////////////////////////////////////////////////////                                                                                                                                                                                                         
    // nested loop (currently 2-levels)                                                                                                                                                                                                                                       
    //                                                                                                                                                                                                                                                                        
    // NOTE 1: loop<n_i, n_j>::body(lambda) corresponds to                                                                                                                                                                                                                    
    //     for (std::size_t i = 0; i <= n_i; ++i)                                                                                                                                                                                                                            
    //       for (std::size_t j = 0; j <= n_j; ++j)                                                                                                                                                                                                                          
    //                 lambda();                                                                                                                                                                                                                                              
    //                                                                                                                                                                                                                                                                        
    // NOTE 2: loop_pow2<n_i, n_j>::body(lambda) corresponds to                                                                                                                                                                                                               
    //     for (std::size_t i = 1; i <= n_i; i *= 2)                                                                                                                                                                                                                         
    //       for (std::size_t j = 1; j <= n_j; j *= 2)                                                                                                                                                                                                                       
    //                 lambda();                                                                                                                                                                                                                                              
    //                                                                                                                                                                                                                                                                        
    //     where n_i and n_j are assumed to be powers of 2                                                                                                                                                                                                                    
    //                                                                                                                                                                                                                                                                        
    // NOTE 3: loop_triangle<n>::body(lambda) corresponds to                                                                                                                                                                                                                  
    //     for (std::size_t i = 0; i <= n; ++i)                                                                                                                                                                                                                              
    //       for (std::size_t j = 0; j <= i; ++j)                                                                                                                                                                                                                            
    //                 lambda();                                                                                                                                                                                                                                              
    //                                                                                                                                                                                                                                                                            
    // NOTE 4: loop_triangle_pow2<n>::body(lambda) corresponds to                                                                                                                                                                                                                 
    //     for (std::size_t i = 1; i <= n; i *= 2)                                                                                                                                                                                                                               
    //       for (std::size_t j = 1; j <= i; j *= 2)                                                                                                                                                                                                                             
    //             lambda();                                                                                                                                                                                                                                                      
    //                                                                                                                                                                                                                                                                            
    //     where n is assumed to be a power of 2                                                                                                                                                                                                                                  
    //                                                                                                                                                                                                                                                                        
    /////////////////////////////////////////////////////////////////                                                                                                                                                                                                         
    namespace
    {
        // file scope: not accessible outside this file                                                                                                                                                                                                                   

        template <typename T, T X>
        struct template_parameter_t
        {
            // the purpose of this data structure is to make template parameters accessible within lambda                                                                                                                                                             
            // functions by passing them as constant expressions via the argument list                                                                                                                                                                                
            static constexpr T value = X;
        };

        template <std::size_t J, std::size_t I>
        struct loop_nest_level_1
        {
            static_assert(J > 0, "loop trip count needs to be > 0");
            static_assert(I >= 0, "loop trip count needs to be >= 0");

            template <typename F>
            static void execute(F func)
            {
                loop_nest_level_1<J - 1, I>::execute(func);

                template_parameter_t<std::size_t, I> i;
                template_parameter_t<std::size_t, J> j;
                func(i, j);
            }
        };

        template <std::size_t I>
        struct loop_nest_level_1<0, I>
        {
            static_assert(I >= 0, "loop trip count needs to be >= 0");

            template <typename F>
            static void execute(F func)
            {
                template_parameter_t<std::size_t, I> i;
                template_parameter_t<std::size_t, 0> j;
                func(i, j);
            }
        };

        template <std::size_t J, std::size_t I>
        struct loop_pow2_nest_level_1
        {
            static_assert(J > 0 && (J & (J - 1)) == 0, "loop trip count is not a power of 2");
            static_assert(I > 0 && (I & (I - 1)) == 0, "loop trip count is not a power of 2");

            template <typename F>
            static void execute(F func)
            {
                loop_pow2_nest_level_1<J / 2, I>::execute(func);

                template_parameter_t<std::size_t, I> i;
                template_parameter_t<std::size_t, J> j;
                func(i, j);
            }
        };

        template <std::size_t I>
        struct loop_pow2_nest_level_1<1, I>
        {
            static_assert(I > 0 && (I & (I - 1)) == 0, "loop trip count is not a power of 2");

            template <typename F>
            static void execute(F func)
            {
                template_parameter_t<std::size_t, I> i;
                template_parameter_t<std::size_t, 1> j;
                func(i, j);
            }
        };
    }

    template <std::size_t I, std::size_t J>
    struct loop
    {
        static_assert(I > 0, "loop trip count needs to be > 0");
        static_assert(J >= 0, "loop trip count needs to be >= 0");

        template <typename F>
        static void body(F func)
        {
                
                loop<I - 1, J>::body(func);
                loop_nest_level_1<J, I>::execute(func);
                
        }
    };

    template <std::size_t J>
    struct loop<0, J>
    {
        static_assert(J >= 0, "loop trip count needs to be >= 0");

        template <typename F>
        static void body(F func)
        {
                loop_nest_level_1<J, 0>::execute(func);
        }
    };

    template <std::size_t I, std::size_t J>
    struct loop_pow2
    {
        static_assert(J > 0 && (J & (J - 1)) == 0, "loop trip count is not a power of 2");
        static_assert(I > 0 && (I & (I - 1)) == 0, "loop trip count is not a power of 2");

        template <typename F>
        static void body(F func)
        {
                loop_pow2<I / 2, J>::body(func);
                loop_pow2_nest_level_1<J, I>::execute(func);
        }
    };

    template <std::size_t J>
    struct loop_pow2<1, J>
    {
        static_assert(J > 0 && (J & (J - 1)) == 0, "loop trip count is not a power of 2");

        template <typename F>
        static void body(F func)
        {
                loop_pow2_nest_level_1<J, 1>::execute(func);
        }
    };

    template <std::size_t N>
    struct loop_triangle
    {
        static_assert(N >= 0, "loop trip count needs to be >= 0");

        template <typename F>
        static void body(F func)
        {
                loop_triangle<N - 1>::body(func);
                loop_nest_level_1<N, N>::execute(func);
        }
    };

    template <>
    struct loop_triangle<0>
    {
        template <typename F>
        static void body(F func)
        {
                loop_nest_level_1<0, 0>::execute(func);
        }
    };

    template <std::size_t N>
    struct loop_triangle_pow2
    {
        static_assert(N > 0 && (N & (N - 1)) == 0, "loop trip count is not a power of 2");

        template <typename F>
        static void body(F func)
        {
                loop_triangle_pow2<N / 2>::body(func);
                loop_pow2_nest_level_1<N, N>::execute(func);
        }
    };

    template <>
    struct loop_triangle_pow2<1>
    {
        template <typename F>
        static void body(F func)
        {
                loop_pow2_nest_level_1<1, 1>::execute(func);
        }
    };
}

#endif