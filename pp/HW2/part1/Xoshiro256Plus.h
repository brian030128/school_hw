#pragma once

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256+ 1.0, our best and fastest generator for floating-point
   numbers. We suggest to use its upper bits for floating-point
   generation, as it is slightly faster than xoshiro256++/xoshiro256**. It
   passes all tests we are aware of except for the lowest three bits,
   which might fail linearity tests (and just those), so if low linear
   complexity is not considered an issue (as it is usually the case) it
   can be used to generate 64-bit outputs, too.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

/*
    Stephan Friedl
    Derived from Public Domain code
    
    Modified to be SIMD-only 8x float generator
*/

/*
    A note on Xoshiro256Plus:

   The statistics on this RNG are very good but if you *need* something for crypto - you may want to look
   for a different RNG. Aside from crypto - this RNG should be perfectly fine.

    Anything is better than the C Lib rand().
*/

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include <array>

#include "SplitMix64.h"

namespace SEFUtility::RNG
{
    class Xoshiro256PlusSIMD8
    {
    public:
        class EightFloatValues
        {
        public:
            EightFloatValues& operator=(EightFloatValues) = delete;
            EightFloatValues& operator=(const EightFloatValues&) = delete;
            EightFloatValues& operator=(EightFloatValues&&) = delete;

            operator __m256() const { return result_packed_; }

            float operator[](size_t index) const 
            { 
                assert(index < 8);
                return reinterpret_cast<const float*>(&result_packed_)[index];
            }

        public:
            alignas(32) __m256 result_packed_;

            EightFloatValues(__m256 value) : result_packed_(value) {}

            EightFloatValues(EightFloatValues&& value_to_copy)
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightFloatValues(const EightFloatValues&) = delete;

            friend class Xoshiro256PlusSIMD8;
        };

        enum class JumpOnCopy : int32_t
        {
            None = 0,
            Short,
            Long
        };

        Xoshiro256PlusSIMD8(const uint64_t seed)
        {
            SplitMix64 split_mix(seed);

            // Initialize 8 independent generator states
            std::array<std::array<uint64_t, 4>, 8> initial_states;
            
            initial_states[0][0] = split_mix.next();
            initial_states[0][1] = split_mix.next();
            initial_states[0][2] = split_mix.next();
            initial_states[0][3] = split_mix.next();

            // Use long jumps to create independent streams
            for (size_t i = 1; i < 8; ++i)
            {
                initial_states[i] = long_jump(initial_states[i - 1]);
            }

            simd_state_ = SIMDState(initial_states);
        }

        Xoshiro256PlusSIMD8(const std::array<std::array<uint64_t, 4>, 8>& seeds)
        {
            simd_state_ = SIMDState(seeds);
        }

        Xoshiro256PlusSIMD8(const Xoshiro256PlusSIMD8& rng_to_copy, JumpOnCopy jump_dist = JumpOnCopy::Short)
            : simd_state_(rng_to_copy.simd_state_, jump_dist)
        {
        }

        // Generate 8 float values in range [0, 1)
        EightFloatValues next8()
        {
            return simd_next8_internal(simd_state_);
        }

        // Generate 8 float values in range [lower_bound, upper_bound)
        EightFloatValues next8(float lower_bound, float upper_bound)
        {
            __m256 values = simd_next8_internal(simd_state_);
            __m256 range = _mm256_set1_ps(upper_bound - lower_bound);
            __m256 lower = _mm256_set1_ps(lower_bound);
            
            // result = values * range + lower_bound
            return _mm256_add_ps(_mm256_mul_ps(values, range), lower);
        }

        // Jump functions for creating independent generator streams
        static std::array<uint64_t, 4> jump(const std::array<uint64_t, 4>& initial_state)
        {
            static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 
                                           0xa9582618e03fc9aa, 0x39abdc4529b1661c};

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

                    next_scalar(local_state);
                }
            }

            return temp;
        }

        static std::array<uint64_t, 4> long_jump(const std::array<uint64_t, 4>& initial_state)
        {
            static const uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 
                                                 0x77710069854ee241, 0x39109bb02acbe635};

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

                    next_scalar(local_state);
                }
            }

            return temp;
        }

    private:
        // Constants for float conversion
        // IEEE 754 float: sign(1) | exponent(8) | mantissa(23)
        // We want to create floats in [1.0, 2.0) then subtract 1.0 to get [0.0, 1.0)
        // 1.0 in float = 0x3F800000 (exponent = 127)
        static constexpr uint32_t FLOAT_MASK = 0x3F800000u;  // 1.0f
        static constexpr uint32_t MANTISSA_MASK = 0x007FFFFFu;  // 23 bits for mantissa

        class alignas(32) SIMDState
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
                        for (size_t i = 0; i < 8; ++i)
                        {
                            uint64_array_state_[i] = jump(uint64_array_state_[i]);
                        }
                        break;

                    case JumpOnCopy::Long:
                        for (size_t i = 0; i < 8; ++i)
                        {
                            uint64_array_state_[i] = long_jump(uint64_array_state_[i]);
                        }
                        break;
                }
            }

            SIMDState(const std::array<std::array<uint64_t, 4>, 8>& seeds)
            {
                for (size_t i = 0; i < 8; ++i)
                {
                    uint64_array_state_[i] = seeds[i];
                }
            }

            const __m256i operator[](size_t index) const 
            { 
                assert(index < 4);
                return packed_state_[index]; 
            }

            __m256i& operator[](size_t index) 
            { 
                assert(index < 4);
                return packed_state_[index]; 
            }

        private:
            union
            {
                __m256i packed_state_[4];
                std::array<std::array<uint64_t, 4>, 4> uint64_array_state_;
            };
        };

        SIMDState simd_state_;

        // Generate 4 uint64 values (in one __m256i)
        static __m256i simd_next4_uint64_internal(SIMDState& state)
        {
            __m256i result = _mm256_add_epi64(state[0], state[3]);
            const __m256i temp = _mm256_slli_epi64(state[1], 17);

            state[2] = _mm256_xor_si256(state[2], state[0]);
            state[3] = _mm256_xor_si256(state[3], state[1]);
            state[1] = _mm256_xor_si256(state[1], state[2]);
            state[0] = _mm256_xor_si256(state[0], state[3]);
            state[2] = _mm256_xor_si256(state[2], temp);
            state[3] = rotl(state[3], 45);

            return result;
        }

        // Generate 8 float values in [0, 1) packed in one __m256
        static EightFloatValues simd_next8_internal(SIMDState& state)
        {
            // Get 8 uint64 values (two SIMD operations)
            __m256i values_low = simd_next4_uint64_internal(state);
            __m256i values_high = simd_next4_uint64_internal(state);

            // Use upper 32 bits from each uint64 for better quality
            // Shift right by 32 to get upper 32 bits
            __m256i upper_low = _mm256_srli_epi64(values_low, 32);
            __m256i upper_high = _mm256_srli_epi64(values_high, 32);

            // Pack 4 uint32 from each into 8 uint32 total
            __m256i shuffled_low = _mm256_shuffle_epi32(upper_low, _MM_SHUFFLE(2, 0, 2, 0));
            __m256i shuffled_high = _mm256_shuffle_epi32(upper_high, _MM_SHUFFLE(2, 0, 2, 0));

            __m128i low_128 = _mm256_castsi256_si128(shuffled_low);
            __m128i high_128 = _mm256_castsi256_si128(shuffled_high);
            
            __m256i packed_uint32 = _mm256_set_m128i(high_128, low_128);

            // Convert to float [0, 1):
            // 1. Take upper 23 bits for mantissa (right shift by 9)
            // 2. OR with 0x3F800000 (creates float in [1.0, 2.0))
            // 3. Subtract 1.0 to get [0.0, 1.0)
            
            __m256i mantissa = _mm256_srli_epi32(packed_uint32, 9);
            __m256i float_bits = _mm256_or_si256(mantissa, _mm256_set1_epi32(FLOAT_MASK));
            
            // Reinterpret as float and subtract 1.0
            __m256 result = _mm256_castsi256_ps(float_bits);
            result = _mm256_sub_ps(result, _mm256_set1_ps(1.0f));

            return result;
        }

        static uint64_t next_scalar(std::array<uint64_t, 4>& state)
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

        static inline uint64_t rotl(const uint64_t x, int k) 
        { 
            return (x << k) | (x >> (64 - k)); 
        }

        static inline __m256i rotl(const __m256i x, int k)
        {
            return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k));
        }
    };

}  // namespace SEFUtility::RNG