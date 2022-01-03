// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#if defined(ARROW_HAVE_AVX2)
#include <immintrin.h>
#endif

#include <cstdint>

#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/util.h"

namespace arrow {
namespace compute {

// Implementations are based on xxh3 32-bit algorithm description from:
// https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
//
class Hashing32 {
 public:
  static void hash_fixed(int64_t hardware_flags, bool combine_hashes, uint32_t num_keys,
                         uint64_t length_key, const uint8_t* keys, uint32_t* hashes,
                         uint32_t* temp_hashes_for_combine);

  static void hash_varlen(int64_t hardware_flags, bool combine_hashes, uint32_t num_rows,
                          const uint32_t* offsets, const uint8_t* concatenated_keys,
                          uint32_t* hashes, uint32_t* temp_hashes_for_combine);

  static void hash_varlen(int64_t hardware_flags, bool combine_hashes, uint32_t num_rows,
                          const uint64_t* offsets, const uint8_t* concatenated_keys,
                          uint32_t* hashes, uint32_t* temp_hashes_for_combine);

  static void HashMultiColumn(const std::vector<KeyEncoder::KeyColumnArray>& cols,
                              KeyEncoder::KeyEncoderContext* ctx, uint32_t* out_hash);

 private:
  static const uint32_t PRIME32_1 = 0x9E3779B1;
  static const uint32_t PRIME32_2 = 0x85EBCA77;
  static const uint32_t PRIME32_3 = 0xC2B2AE3D;
  static const uint32_t PRIME32_4 = 0x27D4EB2F;
  static const uint32_t PRIME32_5 = 0x165667B1;
  static const uint32_t kCombineConst = 0x9e3779b9UL;
  static const int64_t kStripeSize = 4 * sizeof(uint32_t);

  static inline uint32_t avalanche(uint32_t acc) {
    acc ^= (acc >> 15);
    acc *= PRIME32_2;
    acc ^= (acc >> 13);
    acc *= PRIME32_3;
    acc ^= (acc >> 16);
    return acc;
  }
  static inline uint32_t round(uint32_t acc, uint32_t input);
  static inline uint32_t combine_accumulators(uint32_t acc1, uint32_t acc2, uint32_t acc3,
                                              uint32_t acc4);
  static inline uint32_t combine_hashes(uint32_t previous_hash, uint32_t hash) {
    uint32_t next_hash = previous_hash ^ (hash + kCombineConst + (previous_hash << 6) +
                                          (previous_hash >> 2));
    return next_hash;
  }
  static inline void process_full_stripes(uint64_t num_stripes, const uint8_t* key,
                                          uint32_t* out_acc1, uint32_t* out_acc2,
                                          uint32_t* out_acc3, uint32_t* out_acc4);
  static inline void process_last_stripe(uint32_t mask1, uint32_t mask2, uint32_t mask3,
                                         uint32_t mask4, const uint8_t* last_stripe,
                                         uint32_t* acc1, uint32_t* acc2, uint32_t* acc3,
                                         uint32_t* acc4);
  static inline void stripe_mask(int i, uint32_t* mask1, uint32_t* mask2, uint32_t* mask3,
                                 uint32_t* mask4);
  template <bool combine_hashes>
  static void hash_fixedlen_imp(uint32_t num_rows, uint64_t length, const uint8_t* keys,
                                uint32_t* hashes);
  template <typename T, bool combine_hashes>
  static void hash_varlen_imp(uint32_t num_rows, const T* offsets,
                              const uint8_t* concatenated_keys, uint32_t* hashes);
  template <bool combine_hashes>
  static void hash_bit_imp(int64_t bit_offset, uint32_t num_keys, const uint8_t* keys,
                           uint32_t* hashes);
  static void hash_bit(bool combine_hashes, int64_t bit_offset, uint32_t num_keys,
                       const uint8_t* keys, uint32_t* hashes);
  template <bool combine_hashes, typename T>
  static void hash_int_imp(uint32_t num_keys, const T* keys, uint32_t* hashes);
  static void hash_int(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                       const uint8_t* keys, uint32_t* hashes);

#if defined(ARROW_HAVE_AVX2)
  static inline __m256i avalanche_avx2(__m256i hash);
  static inline __m256i combine_hashes_avx2(__m256i previous_hash, __m256i hash);
  template <bool combine_hashes>
  static void avalanche_all_avx2(uint32_t num_rows, uint32_t* hashes,
                                 const uint32_t* hashes_temp_for_combine);
  static inline __m256i round_avx2(__m256i acc, __m256i input);
  static inline uint64_t combine_accumulators_avx2(__m256i acc);
  static inline __m256i stripe_mask_avx2(int i, int j);
  template <bool two_equal_lengths>
  static inline __m256i process_stripes_avx2(int64_t num_stripes_A, int64_t num_stripes_B,
                                             __m256i mask_last_stripe,
                                             const uint8_t* keys, int64_t offset_A,
                                             int64_t offset_B);
  template <bool combine_hashes>
  static uint32_t hash_fixedlen_imp_avx2(uint32_t num_rows, uint64_t length,
                                         const uint8_t* keys, uint32_t* hashes,
                                         uint32_t* hashes_temp_for_combine);
  static uint32_t hash_fixedlen_avx2(bool combine_hashes, uint32_t num_rows,
                                     uint64_t length, const uint8_t* keys,
                                     uint32_t* hashes, uint32_t* hashes_temp_for_combine);
  template <typename T, bool combine_hashes>
  static uint32_t hash_varlen_imp_avx2(uint32_t num_rows, const T* offsets,
                                       const uint8_t* concatenated_keys, uint32_t* hashes,
                                       uint32_t* hashes_temp_for_combine);
  static uint32_t hash_varlen_avx2(bool combine_hashes, uint32_t num_rows,
                                   const uint32_t* offsets,
                                   const uint8_t* concatenated_keys, uint32_t* hashes,
                                   uint32_t* hashes_temp_for_combine);
  static uint32_t hash_varlen_avx2(bool combine_hashes, uint32_t num_rows,
                                   const uint64_t* offsets,
                                   const uint8_t* concatenated_keys, uint32_t* hashes,
                                   uint32_t* hashes_temp_for_combine);
#endif
};

class Hashing64 {
 public:
  static void hash_fixed(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                         const uint8_t* keys, uint64_t* hashes);

  static void hash_varlen(bool combine_hashes, uint32_t num_rows, const uint32_t* offsets,
                          const uint8_t* concatenated_keys, uint64_t* hashes);

  static void hash_varlen(bool combine_hashes, uint32_t num_rows, const uint64_t* offsets,
                          const uint8_t* concatenated_keys, uint64_t* hashes);

  static void HashMultiColumn(const std::vector<KeyEncoder::KeyColumnArray>& cols,
                              KeyEncoder::KeyEncoderContext* ctx, uint64_t* hashes);

 private:
  static const uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
  static const uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
  static const uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
  static const uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
  static const uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;
  static const uint32_t kCombineConst = 0x9e3779b9UL;
  static const int64_t kStripeSize = 4 * sizeof(uint64_t);

  static inline uint64_t avalanche(uint64_t acc);
  static inline uint64_t round(uint64_t acc, uint64_t input);
  static inline uint64_t combine_accumulators(uint64_t acc1, uint64_t acc2, uint64_t acc3,
                                              uint64_t acc4);
  static inline void combine_hashes(uint64_t* multi_column_hash, uint64_t hash);
  static inline void process_full_stripes(uint64_t num_stripes, const uint8_t* key,
                                          uint64_t* out_acc1, uint64_t* out_acc2,
                                          uint64_t* out_acc3, uint64_t* out_acc4);
  static inline void process_last_stripe(uint64_t mask1, uint64_t mask2, uint64_t mask3,
                                         uint64_t mask4, const uint8_t* last_stripe,
                                         uint64_t* acc1, uint64_t* acc2, uint64_t* acc3,
                                         uint64_t* acc4);
  static inline void stripe_mask(int i, uint64_t* mask1, uint64_t* mask2, uint64_t* mask3,
                                 uint64_t* mask4);
  template <bool combine_hashes>
  static void hash_fixedlen_imp(uint32_t num_rows, uint64_t length, const uint8_t* keys,
                                uint64_t* hashes);
  template <typename T, bool combine_hashes>
  static void hash_varlen_imp(uint32_t num_rows, const T* offsets,
                              const uint8_t* concatenated_keys, uint64_t* hashes);
  template <bool combine_hashes>
  static void hash_bit_imp(int64_t bit_offset, uint32_t num_keys, const uint8_t* keys,
                           uint64_t* hashes);
  static void hash_bit(bool combine_hashes, int64_t bit_offset, uint32_t num_keys,
                       const uint8_t* keys, uint64_t* hashes);
  template <bool combine_hashes, typename T>
  static void hash_int_imp(uint32_t num_keys, const T* keys, uint64_t* hashes);
  static void hash_int(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                       const uint8_t* keys, uint64_t* hashes);
};

}  // namespace compute
}  // namespace arrow
