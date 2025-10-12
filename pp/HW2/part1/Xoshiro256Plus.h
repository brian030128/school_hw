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
    
    Modified to be SIMD-only 8x uint32 generator
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
        class EightIntegerValues
        {
        public:
            EightIntegerValues& operator=(EightIntegerValues) = delete;
            EightIntegerValues& operator=(const EightIntegerValues&) = delete;
            EightIntegerValues& operator=(EightIntegerValues&&) = delete;

            operator __m256i() const { return result_packed_; }

            uint32_t operator[](size_t index) const 
            { 
                assert(index < 8);
                return reinterpret_cast<const uint32_t*>(&result_packed_)[index];
            }

        public:
            alignas(32) __m256i result_packed_;

            EightIntegerValues(__m256i value) : result_packed_(value) {}

            EightIntegerValues(EightIntegerValues&& value_to_copy)
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightIntegerValues(const EightIntegerValues&) = delete;

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

        // Generate 8 uint32_t values at once (packed in one __m256i)
        EightIntegerValues next8()
        {
            return simd_next8_internal(simd_state_);
        }

        // Generate 8 bounded uint32_t values in range [lower_bound, upper_bound)
        EightIntegerValues next8(uint32_t lower_bound, uint32_t upper_bound)
        {
            assert(upper_bound > lower_bound);

            uint64_t range = upper_bound - lower_bound;
            
            // Get 8 uint64 values from the two SIMD lanes
            __m256i values_low = simd_next4_uint64_internal(simd_state_);
            __m256i values_high = simd_next4_uint64_internal(simd_state_);

            __m256i range_vec = _mm256_set1_epi64x(range);
            __m256i lower_vec = _mm256_set1_epi64x(lower_bound);

            // Compute bounded values: ((uint32_t(value) * range) >> 32) + lower_bound
            __m256i bounded_low = _mm256_add_epi64(
                _mm256_srli_epi64(_mm256_mul_epu32(values_low, range_vec), 32),
                lower_vec);

            __m256i bounded_high = _mm256_add_epi64(
                _mm256_srli_epi64(_mm256_mul_epu32(values_high, range_vec), 32),
                lower_vec);

            // Pack 4 uint64 -> 4 uint32 for each lane
            // Extract lower 32 bits from each 64-bit value and pack them
            __m256i shuffled_low = _mm256_shuffle_epi32(bounded_low, _MM_SHUFFLE(2, 0, 2, 0));
            __m256i shuffled_high = _mm256_shuffle_epi32(bounded_high, _MM_SHUFFLE(2, 0, 2, 0));

            // Combine: take lower 128 bits from each
            __m128i low_128 = _mm256_castsi256_si128(shuffled_low);
            __m128i high_128 = _mm256_castsi256_si128(shuffled_high);

            // Interleave to get 8 uint32 values in one __m256i
            return _mm256_set_m128i(high_128, low_128);
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

        // Generate 8 uint32 values packed in one __m256i
        static EightIntegerValues simd_next8_internal(SIMDState& state)
        {
            // Get 8 uint64 values (two SIMD operations)
            __m256i values_low = simd_next4_uint64_internal(state);
            __m256i values_high = simd_next4_uint64_internal(state);

            // Extract lower 32 bits from each 64-bit value
            // Shuffle to pack: [a0, _, a1, _, a2, _, a3, _] -> [a0, a1, a2, a3, ...]
            __m256i shuffled_low = _mm256_shuffle_epi32(values_low, _MM_SHUFFLE(2, 0, 2, 0));
            __m256i shuffled_high = _mm256_shuffle_epi32(values_high, _MM_SHUFFLE(2, 0, 2, 0));

            // Pack the two 128-bit halves into one 256-bit register
            __m128i low_128 = _mm256_castsi256_si128(shuffled_low);
            __m128i high_128 = _mm256_castsi256_si128(shuffled_high);

            return _mm256_set_m128i(high_128, low_128);
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