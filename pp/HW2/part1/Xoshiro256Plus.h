#pragma once

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256+ 1.0, our best and fastest generator for floating-point
   numbers. We suggest to use its upper bits for floating-point
   generation, as it is slightly faster than xoshiro256++/xoshiro256**.

   Modified to generate 8 floats at a time using AVX2/AVX-512 instructions.
*/

/*
    Stephan Friedl
    Derived from Public Domain code
    Modified for 8-way float generation
*/

/*
    A note on Xoshiro256Plus:

   The statistics on this RNG are very good but if you *need* something for crypto - you may want to look
   for a different RNG.  Aside from crypto - this RNG should be perfectly fine.

    Anything is better than the C Lib rand().
*/

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include <array>
#include <limits>

#include "SplitMix64.h"

namespace SEFUtility::RNG
{
    template <SIMDInstructionSet SIMD>
    class Xoshiro256Plus
    {
       public:
        class EightIntegerValues
        {
           public:
            EightIntegerValues& operator=(EightIntegerValues) = delete;
            EightIntegerValues& operator=(const EightIntegerValues&) = delete;
            EightIntegerValues& operator=(EightIntegerValues&&) = delete;

#ifdef __AVX512F__
            operator __m512i() const { return result_packed_; }
#endif

            uint64_t operator[](size_t index) const { return result_packed_[index]; }

           public:
            alignas(64) __m512i result_packed_;

            EightIntegerValues(uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4,
                              uint64_t v5, uint64_t v6, uint64_t v7, uint64_t v8)
            {
                if (SIMD >= SIMDInstructionSet::AVX512)
                {
                    result_packed_ = _mm512_set_epi64(v8, v7, v6, v5, v4, v3, v2, v1);
                }
                else
                {
                    result_packed_[0] = v1;
                    result_packed_[1] = v2;
                    result_packed_[2] = v3;
                    result_packed_[3] = v4;
                    result_packed_[4] = v5;
                    result_packed_[5] = v6;
                    result_packed_[6] = v7;
                    result_packed_[7] = v8;
                }
            }

            EightIntegerValues(__m512i value) : result_packed_(std::move(value)) {}

            EightIntegerValues(EightIntegerValues&& value_to_copy)
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightIntegerValues(EightIntegerValues& value_to_copy) = delete;
            EightIntegerValues(const EightIntegerValues& value_to_copy) = delete;

            friend class Xoshiro256Plus;
        };

        class EightFloatValues
        {
           public:
            EightFloatValues& operator=(EightFloatValues) = delete;
            EightFloatValues& operator=(const EightFloatValues&) = delete;
            EightFloatValues& operator=(EightFloatValues&&) = delete;

#ifdef __AVX512F__
            operator __m512() const { return result_packed_; }
#endif

            float operator[](size_t index) const { return result_packed_[index]; }

           public:
            alignas(64) __m512 result_packed_;

#ifdef __AVX512F__
            EightFloatValues(__m512 value) : result_packed_(std::move(value)) {}
#else
            EightFloatValues(__m512& value) : result_packed_(std::move(value)) {}
#endif

            EightFloatValues(EightFloatValues&& value_to_copy) 
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightFloatValues(EightFloatValues& value_to_copy) = delete;
            EightFloatValues(const EightFloatValues& value_to_copy) = delete;

            friend class Xoshiro256Plus;
        };

        enum class JumpOnCopy : int32_t
        {
            None = 0,
            Short,
            Long
        };

        Xoshiro256Plus(const uint64_t seed)
        {
            static_assert(SIMD != SIMDInstructionSet::AVX, "AVX RNG is not supported - just use NONE");

#ifndef __AVX512F__
            static_assert(SIMD == SIMDInstructionSet::NONE,
                          "Cannot have an AVX-512 RNG if AVX-512 extensions are not available");
#endif

            SplitMix64 split_mix(seed);

            serial_state_[0] = split_mix.next();
            serial_state_[1] = split_mix.next();
            serial_state_[2] = split_mix.next();
            serial_state_[3] = split_mix.next();

            // Initialize 8 independent streams for SIMD generation
            serial_next8_state_[0] = long_jump(serial_state_);
            serial_next8_state_[1] = long_jump(serial_next8_state_[0]);
            serial_next8_state_[2] = long_jump(serial_next8_state_[1]);
            serial_next8_state_[3] = long_jump(serial_next8_state_[2]);
            serial_next8_state_[4] = long_jump(serial_next8_state_[3]);
            serial_next8_state_[5] = long_jump(serial_next8_state_[4]);
            serial_next8_state_[6] = long_jump(serial_next8_state_[5]);
            serial_next8_state_[7] = long_jump(serial_next8_state_[6]);

            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                simd_state_ = SIMDState(serial_next8_state_);
            }
        }

        Xoshiro256Plus(const std::array<uint64_t, 4> seed) : serial_state_(seed)
        {
            static_assert(SIMD != SIMDInstructionSet::AVX, "AVX RNG is not supported - just use NONE");

#ifndef __AVX512F__
            static_assert(SIMD == SIMDInstructionSet::NONE,
                          "Cannot have an AVX-512 RNG if AVX-512 extensions are not available");
#endif

            serial_next8_state_[0] = long_jump(serial_state_);
            serial_next8_state_[1] = long_jump(serial_next8_state_[0]);
            serial_next8_state_[2] = long_jump(serial_next8_state_[1]);
            serial_next8_state_[3] = long_jump(serial_next8_state_[2]);
            serial_next8_state_[4] = long_jump(serial_next8_state_[3]);
            serial_next8_state_[5] = long_jump(serial_next8_state_[4]);
            serial_next8_state_[6] = long_jump(serial_next8_state_[5]);
            serial_next8_state_[7] = long_jump(serial_next8_state_[6]);

            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                simd_state_ = SIMDState(serial_next8_state_);
            }
        }

        Xoshiro256Plus(const Xoshiro256Plus<SIMD>& rng_to_copy, JumpOnCopy jump_dist = JumpOnCopy::Short)
            : serial_state_(rng_to_copy.serial_state_),
              serial_next8_state_(rng_to_copy.serial_next8_state_),
              simd_state_(rng_to_copy.simd_state_, jump_dist)
        {
            switch (jump_dist)
            {
                case JumpOnCopy::None:
                    break;

                case JumpOnCopy::Short:
                    serial_state_ = jump(serial_state_);
                    for (int i = 0; i < 8; ++i)
                    {
                        serial_next8_state_[i] = jump(serial_next8_state_[i]);
                    }
                    break;

                case JumpOnCopy::Long:
                    serial_state_ = long_jump(serial_state_);
                    for (int i = 0; i < 8; ++i)
                    {
                        serial_next8_state_[i] = long_jump(serial_next8_state_[i]);
                    }
                    break;
            }
        }

        //
        //  Single uint64 at a time
        //
        //  Bounding is in the range of [lower,upper) - i.e. lower included, upper not
        //

        uint64_t next(void) { return next_internal(serial_state_); }

        uint64_t next(uint32_t lower_bound, uint32_t upper_bound)
        {
            assert(upper_bound > lower_bound);

            return (((uint64_t)((uint32_t)next()) * (uint64_t)(upper_bound - lower_bound)) >> 32) +
                   (uint64_t)lower_bound;
        }

        //
        //  Eight uint64s at a time
        //
        //  Bounding is in the range of [lower,upper) - i.e. lower included, upper not
        //

        EightIntegerValues next8()
        {
            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                return simd_next8_internal(simd_state_);
            }
            else
            {
                return EightIntegerValues(
                    next_internal(serial_next8_state_[0]), next_internal(serial_next8_state_[1]),
                    next_internal(serial_next8_state_[2]), next_internal(serial_next8_state_[3]),
                    next_internal(serial_next8_state_[4]), next_internal(serial_next8_state_[5]),
                    next_internal(serial_next8_state_[6]), next_internal(serial_next8_state_[7]));
            }
        }

        EightIntegerValues next8(uint32_t lower_bound, uint32_t upper_bound)
        {
            assert(upper_bound > lower_bound);

            uint64_t range = upper_bound - lower_bound;

            auto eight_ints = next8();

            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                return _mm512_add_epi64(
                    _mm512_srli_epi64(_mm512_mullo_epi32(eight_ints, _mm512_set1_epi64(range)), 32),
                    _mm512_set1_epi64(lower_bound));
            }
            else
            {
                for (int i = 0; i < 8; ++i)
                {
                    eight_ints.result_packed_[i] =
                        (((uint64_t)((uint32_t)eight_ints[i]) * range) >> 32) + (uint64_t)lower_bound;
                }
                return eight_ints;
            }
        }

        //
        //  Single float in range [0,1] for default or [lower, upper] when bounds applied
        //

        float fnext(void)
        {
            union
            {
                uint32_t int_value;
                float float_value;
            };

            int_value = ((uint32_t)(next() >> 32) >> 9) | FLOAT_MASK;

            return float_value - 1.0f;
        }

        float fnext(float lower_bound, float upper_bound)
        {
            return (fnext() * (upper_bound - lower_bound)) + lower_bound;
        }

        //
        //  Eight floats at a time - same bounding as single float
        //

        EightFloatValues fnext8()
        {
            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                union
                {
                    __m512i result_packed_int;
                    __m512 result_packed_float;
                };

                // Get 8 random 64-bit integers
                auto random_ints = next8();
                
                // Convert to 32-bit by shifting right and extracting upper 32 bits
                // Then shift right by 9 to get 23 bits for mantissa
                __m512i shifted = _mm512_srli_epi64(random_ints, 32);
                __m512i as_32bit = _mm512_cvtepi64_epi32(shifted);
                __m512i mantissa_bits = _mm512_srli_epi32(as_32bit, 9);
                
                // OR with float mask to create [1.0, 2.0) range
                result_packed_int = _mm512_or_si512(_mm512_cvtepi32_epi64(mantissa_bits), 
                                                     _mm512_set1_epi64(FLOAT_MASK_64));

                // Convert to float and subtract 1.0 to get [0.0, 1.0)
                __m512i low_32 = _mm512_cvtepi64_epi32(result_packed_int);
                result_packed_float = _mm512_castsi512_ps(low_32);
                
                return _mm512_sub_ps(result_packed_float, _mm512_set1_ps(1.0f));
            }
            else
            {
                union
                {
                    uint32_t int_value;
                    float float_value;
                };

                __m512 packed_result;

                for (int i = 0; i < 8; ++i)
                {
                    int_value = ((uint32_t)(next_internal(serial_next8_state_[i]) >> 32) >> 9) | FLOAT_MASK;
                    packed_result[i] = float_value - 1.0f;
                }

                return packed_result;
            }
        }

        EightFloatValues fnext8(float lower_bound, float upper_bound)
        {
            if constexpr (SIMD >= SIMDInstructionSet::AVX512)
            {
                __m512 range = _mm512_set1_ps(upper_bound - lower_bound);
                __m512 lower = _mm512_set1_ps(lower_bound);
                return _mm512_add_ps(_mm512_mul_ps(fnext8(), range), lower);
            }
            else
            {
                auto result = fnext8();
                float range = upper_bound - lower_bound;

                for (int i = 0; i < 8; ++i)
                {
                    result.result_packed_[i] = (result.result_packed_[i] * range) + lower_bound;
                }
            
                return result;
            }
        }

        //
        //  Jump Functions
        //

        //  This is the jump function for the generator. It is equivalent
        //     to 2^128 calls to next(); it can be used to generate 2^128
        //     non-overlapping subsequences for parallel computations.

        static std::array<uint64_t, 4> jump(const std::array<uint64_t, 4>& initial_state)
        {
            static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa,
                                            0x39abdc4529b1661c};

            std::array<uint64_t, 4> local_state(initial_state);
            std::array<uint64_t, 4> temp({0, 0, 0, 0});

            for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            {
                for (int b = 0; b < 64; b++)
                {
                    if (JUMP[i] & UINT64_C(1) << b)
                    {
                        temp[0] ^= local_state[0];
                        temp[1] ^= local_state[1];
                        temp[2] ^= local_state[2];
                        temp[3] ^= local_state[3];
                    }

                    next_internal(local_state);
                }
            }

            return temp;
        }

        //  This is the long-jump function for the generator. It is equivalent to
        //      2^192 calls to next(); it can be used to generate 2^64 starting points,
        //      from each of which jump() will generate 2^64 non-overlapping
        //      subsequences for parallel distributed computations.

        static std::array<uint64_t, 4> long_jump(const std::array<uint64_t, 4>& initial_state)
        {
            static const uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                                 0x39109bb02acbe635};

            std::array<uint64_t, 4> local_state(initial_state);
            std::array<uint64_t, 4> temp({0, 0, 0, 0});

            for (int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            {
                for (int b = 0; b < 64; b++)
                {
                    if (LONG_JUMP[i] & UINT64_C(1) << b)
                    {
                        temp[0] ^= local_state[0];
                        temp[1] ^= local_state[1];
                        temp[2] ^= local_state[2];
                        temp[3] ^= local_state[3];
                    }

                    next_internal(local_state);
                }
            }

            return temp;
        }

       private:
        static constexpr uint32_t FLOAT_MASK = UINT32_C(0x3F8) << 20;  // 0x3F800000 for [1.0, 2.0)
        static constexpr uint64_t FLOAT_MASK_64 = UINT64_C(0x3F800000);

        typedef std::array<uint64_t, 4> SerialState;

        alignas(64) SerialState serial_state_;

        alignas(64) std::array<SerialState, 8> serial_next8_state_;

        static inline constexpr __m512 cnstexpr_mm512_set1_ps(float value)
        {
            return (__m512){value, value, value, value, value, value, value, value,
                           value, value, value, value, value, value, value, value};
        };

        static inline constexpr __m512i cnstexpr_mm512_set1_epi64(int64_t value)
        {
            return (__m512i)(__v8di){value, value, value, value, value, value, value, value};
        };

        static constexpr __m512i ONE_PACKED_INT64 = cnstexpr_mm512_set1_epi64(1);
        static constexpr __m512i ZERO_PACKED_INT64 = cnstexpr_mm512_set1_epi64(0);
        static constexpr __m512 ONE_PACKED_FLOAT = cnstexpr_mm512_set1_ps(1.0f);

#ifdef __AVX512F__

        class alignas(64) SIMDState
        {
           public:
            SIMDState() {}

            SIMDState(const SIMDState& state_to_copy, JumpOnCopy jump_dist = JumpOnCopy::None)
                : uint64_array_state_(state_to_copy.uint64_array_state_)
            {
                switch (jump_dist)
                {
                    case JumpOnCopy::None:
                        break;

                    case JumpOnCopy::Short:
                        for (int i = 0; i < 8; ++i)
                        {
                            uint64_array_state_[i] = jump(uint64_array_state_[i]);
                        }
                        break;

                    case JumpOnCopy::Long:
                        for (int i = 0; i < 8; ++i)
                        {
                            uint64_array_state_[i] = long_jump(uint64_array_state_[i]);
                        }
                        break;
                }
            }

            SIMDState(const std::array<SerialState, 8>& state) 
                : SIMDState(state[0], state[1], state[2], state[3], 
                           state[4], state[5], state[6], state[7]) {}

            SIMDState(const std::array<uint64_t, 4>& seed1, const std::array<uint64_t, 4>& seed2,
                      const std::array<uint64_t, 4>& seed3, const std::array<uint64_t, 4>& seed4,
                      const std::array<uint64_t, 4>& seed5, const std::array<uint64_t, 4>& seed6,
                      const std::array<uint64_t, 4>& seed7, const std::array<uint64_t, 4>& seed8)
            {
                // Pack 8 RNG states into 4 AVX-512 registers
                // Each register holds one component (s0, s1, s2, or s3) from all 8 generators
                packed_state_[0] = _mm512_set_epi64(seed8[0], seed7[0], seed6[0], seed5[0],
                                                     seed4[0], seed3[0], seed2[0], seed1[0]);
                packed_state_[1] = _mm512_set_epi64(seed8[1], seed7[1], seed6[1], seed5[1],
                                                     seed4[1], seed3[1], seed2[1], seed1[1]);
                packed_state_[2] = _mm512_set_epi64(seed8[2], seed7[2], seed6[2], seed5[2],
                                                     seed4[2], seed3[2], seed2[2], seed1[2]);
                packed_state_[3] = _mm512_set_epi64(seed8[3], seed7[3], seed6[3], seed5[3],
                                                     seed4[3], seed3[3], seed2[3], seed1[3]);
            }

            const __m512i operator[](size_t index) const { return packed_state_[index]; }
            __m512i& operator[](size_t index) { return packed_state_[index]; }

           private:
            union
            {
                __m512i packed_state_[4];
                std::array<std::array<uint64_t, 8>, 4> uint64_array_state_;
            };
        };

        static EightIntegerValues simd_next8_internal(SIMDState& state)
        {
            EightIntegerValues result(_mm512_add_epi64(state[0], state[3]));

            const __m512i temp = _mm512_slli_epi64(state[1], 17);

            state[2] = _mm512_xor_si512(state[2], state[0]);
            state[3] = _mm512_xor_si512(state[3], state[1]);
            state[1] = _mm512_xor_si512(state[1], state[2]);
            state[0] = _mm512_xor_si512(state[0], state[3]);

            state[2] = _mm512_xor_si512(state[2], temp);

            state[3] = rotl(state[3], 45);

            return result;
        }

#else
        class SIMDState
        {
           public:
            SIMDState() {}
            SIMDState(const SIMDState& state_to_copy, JumpOnCopy jump_dist = JumpOnCopy::None) {}
            SIMDState(const std::array<SerialState, 8>& state) {}
        };
#endif

        SIMDState simd_state_;

        static uint64_t next_internal(SerialState& state)
        {
            const uint64_t result = state[0] + state[3];

            const uint64_t t = state[1] << 17;

            state[2] ^= state[0];
            state[3] ^= state[1];
            state[1] ^= state[2];
            state[0] ^= state[3];

            state[2] ^= t;

            state[3] = rotl(state[3], 45);

            return result;
        }

        static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
        
        static inline __m512i rotl(const __m512i x, int k)
        {
            return _mm512_or_si512(_mm512_slli_epi64(x, k), _mm512_srli_epi64(x, 64 - k));
        }
    };
}  // namespace SEFUtility::RNG