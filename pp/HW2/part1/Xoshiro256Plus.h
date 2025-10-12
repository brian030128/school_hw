#pragma once

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256+ 1.0, our best and fastest generator for floating-point
   numbers. We suggest to use its upper bits for floating-point
   generation, as it is slightly faster than xoshiro256++/xoshiro256**.
   
   Modified to generate 8 floats at a time using AVX2 instructions.
*/

/*
    Stephan Friedl
    Derived from Public Domain code
    Modified for 8-float generation
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
        class EightFloatValues
        {
           public:
            EightFloatValues& operator=(EightFloatValues) = delete;
            EightFloatValues& operator=(const EightFloatValues&) = delete;
            EightFloatValues& operator=(EightFloatValues&&) = delete;

#ifdef __AVX2_AVAILABLE__
            operator __m256() const { return result_packed_; }
#endif

            float operator[](size_t index) const { return result_packed_[index]; }

           public:
            alignas(32) __m256 result_packed_;

#ifdef __AVX2_AVAILABLE__
            EightFloatValues(__m256 value) : result_packed_(std::move(value)) {}
#else
            EightFloatValues(__m256& value) : result_packed_(std::move(value)) {}
#endif

            EightFloatValues(EightFloatValues&& value_to_copy) 
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightFloatValues(EightFloatValues& value_to_copy) = delete;
            EightFloatValues(const EightFloatValues& value_to_copy) = delete;

            friend class Xoshiro256Plus;
        };

        class EightIntegerValues
        {
           public:
            EightIntegerValues& operator=(EightIntegerValues) = delete;
            EightIntegerValues& operator=(const EightIntegerValues&) = delete;
            EightIntegerValues& operator=(EightIntegerValues&&) = delete;

#ifdef __AVX2_AVAILABLE__
            operator __m256i() const { return result_packed_; }
#endif

            uint32_t operator[](size_t index) const 
            { 
                uint32_t result[8];
                _mm256_storeu_si256((__m256i*)result, result_packed_);
                return result[index];
            }

           public:
            alignas(32) __m256i result_packed_;

            EightIntegerValues(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
                             uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7)
            {
                if (SIMD >= SIMDInstructionSet::AVX2)
                {
                    result_packed_ = _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
                }
                else
                {
                    uint32_t temp[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                    result_packed_ = _mm256_loadu_si256((__m256i*)temp);
                }
            }

            EightIntegerValues(__m256i value) : result_packed_(std::move(value)) {}

            EightIntegerValues(EightIntegerValues&& value_to_copy)
                : result_packed_(std::move(value_to_copy.result_packed_))
            {
            }

            EightIntegerValues(EightIntegerValues& value_to_copy) = delete;
            EightIntegerValues(const EightIntegerValues& value_to_copy) = delete;

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

#ifndef __AVX2_AVAILABLE__
            static_assert(SIMD == SIMDInstructionSet::NONE,
                          "Cannot have an AVX2 RNG if AVX2 extensions are not available");
#endif

            SplitMix64 split_mix(seed);

            serial_state_[0] = split_mix.next();
            serial_state_[1] = split_mix.next();
            serial_state_[2] = split_mix.next();
            serial_state_[3] = split_mix.next();

            // Initialize 8 separate RNG states for SIMD generation
            for (int i = 0; i < 8; i++)
            {
                serial_next8_state_[i] = long_jump(i == 0 ? serial_state_ : serial_next8_state_[i - 1]);
            }

            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
            {
                simd_state_ = SIMDState(serial_next8_state_);
            }
        }

        Xoshiro256Plus(const std::array<uint64_t, 4> seed) : serial_state_(seed)
        {
            static_assert(SIMD != SIMDInstructionSet::AVX, "AVX RNG is not supported - just use NONE");

#ifndef __AVX2_AVAILABLE__
            static_assert(SIMD == SIMDInstructionSet::NONE,
                          "Cannot have an AVX2 RNG if AVX2 extensions are not available");
#endif

            // Initialize 8 separate RNG states for SIMD generation
            for (int i = 0; i < 8; i++)
            {
                serial_next8_state_[i] = long_jump(i == 0 ? serial_state_ : serial_next8_state_[i - 1]);
            }

            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
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
                    for (int i = 0; i < 8; i++)
                    {
                        serial_next8_state_[i] = jump(serial_next8_state_[i]);
                    }
                    break;

                case JumpOnCopy::Long:
                    serial_state_ = long_jump(serial_state_);
                    for (int i = 0; i < 8; i++)
                    {
                        serial_next8_state_[i] = long_jump(serial_next8_state_[i]);
                    }
                    break;
            }
        }

        //
        //  Single uint64 at a time
        //

        uint64_t next(void) { return next_internal(serial_state_); }

        uint64_t next(uint32_t lower_bound, uint32_t upper_bound)
        {
            assert(upper_bound > lower_bound);
            return (((uint64_t)((uint32_t)next()) * (uint64_t)(upper_bound - lower_bound)) >> 32) +
                   (uint64_t)lower_bound;
        }

        //
        //  Eight uint32s at a time
        //

        EightIntegerValues next8()
        {
            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
            {
                return simd_next8_internal(simd_state_);
            }
            else
            {
                uint32_t values[8];
                for (int i = 0; i < 8; i++)
                {
                    values[i] = (uint32_t)next_internal(serial_next8_state_[i]);
                }
                return _mm256_loadu_si256((__m256i*)values);
            }
        }

        EightIntegerValues next8(uint32_t lower_bound, uint32_t upper_bound)
        {
            assert(upper_bound > lower_bound);
            uint32_t range = upper_bound - lower_bound;

            auto eight_ints = next8();

            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
            {
                // For each 32-bit value: (value * range) >> 32 + lower_bound
                __m256i range_vec = _mm256_set1_epi32(range);
                __m256i lower_vec = _mm256_set1_epi32(lower_bound);
                
                // Multiply and shift for bounded range
                __m256i low = _mm256_mullo_epi32(eight_ints, range_vec);
                __m256i high = _mm256_mulhi_epu32(eight_ints, range_vec);
                
                // We want the high 32 bits of the 64-bit product
                __m256i result = _mm256_add_epi32(high, lower_vec);
                
                return result;
            }
            else
            {
                uint32_t temp[8];
                _mm256_storeu_si256((__m256i*)temp, eight_ints);
                
                for (int i = 0; i < 8; i++)
                {
                    temp[i] = (uint32_t)((((uint64_t)temp[i] * (uint64_t)range) >> 32) + lower_bound);
                }
                
                return _mm256_loadu_si256((__m256i*)temp);
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
            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
            {
                // Get 8 random 32-bit integers
                __m256i random_ints = next8();
                
                // Shift right by 9 bits and OR with float mask
                random_ints = _mm256_srli_epi32(random_ints, 9);
                random_ints = _mm256_or_si256(random_ints, FLOAT_MASK_PACKED);
                
                // Reinterpret as float and subtract 1.0
                __m256 result = _mm256_castsi256_ps(random_ints);
                result = _mm256_sub_ps(result, ONE_PACKED_FLOAT);
                
                return result;
            }
            else
            {
                union
                {
                    uint32_t int_value;
                    float float_value;
                };

                float temp[8];
                for (int i = 0; i < 8; i++)
                {
                    int_value = ((uint32_t)(next_internal(serial_next8_state_[i]) >> 32) >> 9) | FLOAT_MASK;
                    temp[i] = float_value - 1.0f;
                }

                return _mm256_loadu_ps(temp);
            }
        }

        EightFloatValues fnext8(float lower_bound, float upper_bound)
        {
            if constexpr (SIMD >= SIMDInstructionSet::AVX2)
            {
                __m256 range = _mm256_set1_ps(upper_bound - lower_bound);
                __m256 lower = _mm256_set1_ps(lower_bound);
                return _mm256_add_ps(_mm256_mul_ps(fnext8(), range), lower);
            }
            else
            {
                auto result = fnext8();
                float temp[8];
                _mm256_storeu_ps(temp, result);
                
                for (int i = 0; i < 8; i++)
                {
                    temp[i] = (temp[i] * (upper_bound - lower_bound)) + lower_bound;
                }
                
                return _mm256_loadu_ps(temp);
            }
        }

        //
        //  Jump Functions
        //

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
        static constexpr uint32_t FLOAT_MASK = UINT32_C(0x3F8) << 20;  // Float [1,2) mask

        typedef std::array<uint64_t, 4> SerialState;

        alignas(32) SerialState serial_state_;
        alignas(32) std::array<SerialState, 8> serial_next8_state_;

        static inline constexpr __m256 cnstexpr_mm256_set1_ps(float value)
        {
            return (__m256){value, value, value, value, value, value, value, value};
        }

        static inline constexpr __m256i cnstexpr_mm256_set1_epi32(int32_t value)
        {
            return (__m256i)(__v8si){value, value, value, value, value, value, value, value};
        }

        static constexpr __m256i FLOAT_MASK_PACKED = cnstexpr_mm256_set1_epi32(FLOAT_MASK);
        static constexpr __m256 ONE_PACKED_FLOAT = cnstexpr_mm256_set1_ps(1.0f);

#ifdef __AVX2_AVAILABLE__

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
                        for (int i = 0; i < 8; i++)
                        {
                            uint64_array_state_[i] = jump(uint64_array_state_[i]);
                        }
                        break;

                    case JumpOnCopy::Long:
                        for (int i = 0; i < 8; i++)
                        {
                            uint64_array_state_[i] = long_jump(uint64_array_state_[i]);
                        }
                        break;
                }
            }

            SIMDState(const std::array<SerialState, 8>& states)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        packed_state_[j][i] = states[i][j];
                    }
                }
            }

            const __m256i operator[](size_t index) const { return packed_state_[index]; }
            __m256i& operator[](size_t index) { return packed_state_[index]; }

           private:
            union
            {
                __m256i packed_state_[4];  // 4 x __m256i, each holding 4 uint64s = 8 parallel states
                std::array<std::array<uint64_t, 8>, 4> uint64_array_state_;
            };
        };

        static EightIntegerValues simd_next8_internal(SIMDState& state)
        {
            // Each __m256i holds 4 uint64s, we're running 4 parallel xoshiro256+ generators
            // But we want 8 uint32 outputs, so we'll take high 32 bits from 8 of the uint64s
            
            __m256i result0 = _mm256_add_epi64(state[0], state[3]);
            
            const __m256i temp = _mm256_slli_epi64(state[1], 17);

            state[2] = _mm256_xor_si256(state[2], state[0]);
            state[3] = _mm256_xor_si256(state[3], state[1]);
            state[1] = _mm256_xor_si256(state[1], state[2]);
            state[0] = _mm256_xor_si256(state[0], state[3]);

            state[2] = _mm256_xor_si256(state[2], temp);
            state[3] = rotl(state[3], 45);

            // Extract high 32 bits from each uint64 and pack into 8 uint32s
            // We have 4 uint64s in result0, extract their high 32 bits
            __m256i shuffled = _mm256_shuffle_epi32(result0, _MM_SHUFFLE(3, 1, 3, 1));
            
            // Pack 4 high32s from first call with 4 more from continuing the sequence
            // For simplicity, we'll use the 4 uint64 results and extract 8 uint32s
            // by using both low and high parts
            
            // Better approach: convert uint64 results to uint32 by taking upper bits
            __m256i mask_high = _mm256_set1_epi64x(0xFFFFFFFF00000000ULL);
            __m256i high_bits = _mm256_and_si256(result0, mask_high);
            __m256i shifted = _mm256_srli_epi64(high_bits, 32);
            
            // Now pack these into a single __m256i of 8 uint32s
            // This requires some creative shuffling
            return _mm256_castsi256_si256(_mm256_permutevar8x32_epi32(
                _mm256_blend_epi32(result0, _mm256_slli_epi64(result0, 32), 0xAA),
                _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0)
            ));
        }

#else
        class SIMDState
        {
           public:
            SIMDState() {}
            SIMDState(const SIMDState& state_to_copy, JumpOnCopy jump_dist = JumpOnCopy::None) {}
            SIMDState(const std::array<SerialState, 8>& states) {}
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
        
        static inline __m256i rotl(const __m256i x, int k)
        {
            return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k));
        }
    };
}  // namespace SEFUtility::RNG