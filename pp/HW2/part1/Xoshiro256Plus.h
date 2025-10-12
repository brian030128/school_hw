#pragma once

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256+ 1.0, optimized for 8 parallel float generation */

/*
    Stephan Friedl
    Derived from Public Domain code
    
    Modified to be SIMD-only 8x float generator (optimized)
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

            std::array<std::array<uint64_t, 4>, 8> initial_states;
            
            initial_states[0][0] = split_mix.next();
            initial_states[0][1] = split_mix.next();
            initial_states[0][2] = split_mix.next();
            initial_states[0][3] = split_mix.next();

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
        inline EightFloatValues next8()
        {
            return simd_next8_internal(simd_state_);
        }

        // Generate 8 float values in range [lower_bound, upper_bound)
        inline EightFloatValues next8(float lower_bound, float upper_bound)
        {
            __m256 values = simd_next8_internal(simd_state_);
            __m256 range = _mm256_set1_ps(upper_bound - lower_bound);
            __m256 lower = _mm256_set1_ps(lower_bound);
            
            return _mm256_fmadd_ps(values, range, lower);  // FMA: values * range + lower
        }

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
        static constexpr uint32_t FLOAT_MASK = 0x3F800000u;

        class alignas(64) SIMDState  // 64-byte align for cache line
        {
        public:
            SIMDState() 
            {
                for (int i = 0; i < 8; ++i)
                {
                    s_[i] = _mm256_setzero_si256();
                }
            }

            SIMDState(const SIMDState& state_to_copy, JumpOnCopy jump_dist = JumpOnCopy::None)
            {
                for (int i = 0; i < 8; ++i)
                {
                    s_[i] = state_to_copy.s_[i];
                }

                if (jump_dist != JumpOnCopy::None)
                {
                    std::array<std::array<uint64_t, 4>, 8> states;
                    
                    for (int gen = 0; gen < 8; ++gen)
                    {
                        for (int component = 0; component < 4; ++component)
                        {
                            states[gen][component] = reinterpret_cast<const uint64_t*>(&s_[component * 2 + gen / 4])[gen % 4];
                        }
                        
                        states[gen] = (jump_dist == JumpOnCopy::Short) ? jump(states[gen]) : long_jump(states[gen]);
                    }
                    
                    for (int component = 0; component < 4; ++component)
                    {
                        s_[component * 2] = _mm256_set_epi64x(
                            states[3][component], states[2][component],
                            states[1][component], states[0][component]
                        );
                        s_[component * 2 + 1] = _mm256_set_epi64x(
                            states[7][component], states[6][component],
                            states[5][component], states[4][component]
                        );
                    }
                }
            }

            SIMDState(const std::array<std::array<uint64_t, 4>, 8>& seeds)
            {
                // Layout: s_[0]=s0_low, s_[1]=s0_high, s_[2]=s1_low, s_[3]=s1_high, etc.
                for (int component = 0; component < 4; ++component)
                {
                    s_[component * 2] = _mm256_set_epi64x(
                        seeds[3][component], seeds[2][component],
                        seeds[1][component], seeds[0][component]
                    );
                    s_[component * 2 + 1] = _mm256_set_epi64x(
                        seeds[7][component], seeds[6][component],
                        seeds[5][component], seeds[4][component]
                    );
                }
            }

            __m256i s_[8];  // s0_low, s0_high, s1_low, s1_high, s2_low, s2_high, s3_low, s3_high
        };

        SIMDState simd_state_;

        static inline EightFloatValues simd_next8_internal(SIMDState& s)
        {
            // Compute results for both groups
            __m256i result_low = _mm256_add_epi64(s.s_[0], s.s_[6]);   // s0_low + s3_low
            __m256i result_high = _mm256_add_epi64(s.s_[1], s.s_[7]);  // s0_high + s3_high

            // Compute temps
            const __m256i t_low = _mm256_slli_epi64(s.s_[2], 17);   // s1_low << 17
            const __m256i t_high = _mm256_slli_epi64(s.s_[3], 17);  // s1_high << 17

            // Update state for both groups in parallel
            s.s_[4] = _mm256_xor_si256(s.s_[4], s.s_[0]);  // s2_low ^= s0_low
            s.s_[5] = _mm256_xor_si256(s.s_[5], s.s_[1]);  // s2_high ^= s0_high
            
            s.s_[6] = _mm256_xor_si256(s.s_[6], s.s_[2]);  // s3_low ^= s1_low
            s.s_[7] = _mm256_xor_si256(s.s_[7], s.s_[3]);  // s3_high ^= s1_high
            
            s.s_[2] = _mm256_xor_si256(s.s_[2], s.s_[4]);  // s1_low ^= s2_low
            s.s_[3] = _mm256_xor_si256(s.s_[3], s.s_[5]);  // s1_high ^= s2_high
            
            s.s_[0] = _mm256_xor_si256(s.s_[0], s.s_[6]);  // s0_low ^= s3_low
            s.s_[1] = _mm256_xor_si256(s.s_[1], s.s_[7]);  // s0_high ^= s3_high

            s.s_[4] = _mm256_xor_si256(s.s_[4], t_low);    // s2_low ^= t_low
            s.s_[5] = _mm256_xor_si256(s.s_[5], t_high);   // s2_high ^= t_high

            s.s_[6] = rotl(s.s_[6], 45);  // s3_low = rotl(s3_low, 45)
            s.s_[7] = rotl(s.s_[7], 45);  // s3_high = rotl(s3_high, 45)

            // Fast conversion to float using permute instructions
            // Extract high 32 bits and convert to float directly
            __m256i permute_mask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
            __m256i packed_low = _mm256_permutevar8x32_epi32(result_low, permute_mask);
            __m256i packed_high = _mm256_permutevar8x32_epi32(result_high, permute_mask);
            
            // Blend to interleave
            __m256i packed = _mm256_blend_epi32(packed_low, _mm256_slli_si256(packed_high, 4), 0xAA);
            
            // Convert to float [0, 1)
            __m256i mantissa = _mm256_srli_epi32(packed, 9);
            __m256i float_bits = _mm256_or_si256(mantissa, _mm256_set1_epi32(FLOAT_MASK));
            
            __m256 result = _mm256_castsi256_ps(float_bits);
            return _mm256_sub_ps(result, _mm256_set1_ps(1.0f));
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