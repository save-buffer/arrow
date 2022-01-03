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

#include <gmock/gmock-matchers.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <set>
#include <thread>
#include "arrow/compute/exec/bloom_filter.h"
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/cpu_info.h"

namespace arrow {
namespace compute {

TEST(BloomFilter, Masks) {
  BloomFilterMasks masks;
  std::vector<int> sum_num_masks_with_same_n_bits(BloomFilterMasks::kMaxBitsSet + 1);
  std::vector<int> num_masks_with_same_n_bits(BloomFilterMasks::kMaxBitsSet + 1);

  for (bool with_rotation : {false, true}) {
    printf("With bit rotation: %s\n", with_rotation ? "ON" : "OFF");

    for (int i = 0; i <= BloomFilterMasks::kMaxBitsSet; ++i) {
      sum_num_masks_with_same_n_bits[i] = 0;
    }

    for (int imask = 0; imask < BloomFilterMasks::kNumMasks; ++imask) {
      uint64_t mask = masks.mask(imask);
      // Verify that the number of bits set is in the required range
      //
      ARROW_DCHECK(ARROW_POPCOUNT64(mask) >= BloomFilterMasks::kMinBitsSet &&
                   ARROW_POPCOUNT64(mask) <= BloomFilterMasks::kMaxBitsSet);

      for (int i = 0; i <= BloomFilterMasks::kMaxBitsSet; ++i) {
        num_masks_with_same_n_bits[i] = 0;
      }
      for (int imask2nd = 0; imask2nd < BloomFilterMasks::kNumMasks; ++imask2nd) {
        if (imask == imask2nd) {
          continue;
        }
        uint64_t mask_to_compare_to = masks.mask(imask2nd);
        for (int bits_to_rotate = 0; bits_to_rotate < (with_rotation ? 64 : 1);
             ++bits_to_rotate) {
          uint64_t mask_rotated = bits_to_rotate == 0
                                      ? mask_to_compare_to
                                      : ROTL64(mask_to_compare_to, bits_to_rotate);
          ++num_masks_with_same_n_bits[ARROW_POPCOUNT64(mask & mask_rotated)];
        }
      }
      for (int i = 0; i <= BloomFilterMasks::kMaxBitsSet; ++i) {
        sum_num_masks_with_same_n_bits[i] += num_masks_with_same_n_bits[i];
      }
    }

    printf(
        "Expected fraction of masks with the same N bits as any random "
        "mask:\n");
    for (int i = 0; i <= BloomFilterMasks::kMaxBitsSet; ++i) {
      printf("%d. %.2f \n", i,
             static_cast<float>(sum_num_masks_with_same_n_bits[i]) /
                 (static_cast<float>(BloomFilterMasks::kNumMasks *
                                     BloomFilterMasks::kNumMasks) *
                  (with_rotation ? 64 : 1)));
    }
    printf("\n");
  }
}

Status BuildBloomFilter(BloomFilterBuildStrategy strategy, size_t num_threads,
                        int64_t hardware_flags, MemoryPool* pool, int64_t num_rows,
                        std::function<void(int64_t, int, uint32_t*)> get_hash32_impl,
                        std::function<void(int64_t, int, uint64_t*)> get_hash64_impl,
                        BlockedBloomFilter* target, float* build_cost) {
  constexpr int batch_size_max = 32 * 1024;
  int64_t num_batches = bit_util::CeilDiv(num_rows, batch_size_max);

  // omp_set_num_threads(static_cast<int>(num_threads));

  auto builder = BloomFilterBuilder::Make(strategy);

  std::vector<std::vector<uint32_t>> thread_local_hashes32;
  std::vector<std::vector<uint64_t>> thread_local_hashes64;
  thread_local_hashes32.resize(num_threads);
  thread_local_hashes64.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    thread_local_hashes32[i].resize(batch_size_max);
    thread_local_hashes64[i].resize(batch_size_max);
  }

  std::vector<float> build_cost_vector;
  int64_t num_repeats =
      std::max(static_cast<int64_t>(1), bit_util::CeilDiv(1LL << 27, num_rows));
#ifndef NDEBUG
  num_repeats = 1LL;
#endif
  build_cost_vector.resize(num_repeats);

  for (int64_t irepeat = 0; irepeat < num_repeats; ++irepeat) {
    auto time0 = std::chrono::high_resolution_clock::now();

    RETURN_NOT_OK(builder->Begin(num_threads, hardware_flags, pool, num_rows,
                                 bit_util::CeilDiv(num_rows, batch_size_max), target));

    // #pragma omp parallel for
    for (int64_t i = 0; i < builder->num_tasks(); ++i) {
      builder->RunInitTask(i);
    }

    // #pragma omp parallel for
    for (int64_t i = 0; i < num_batches; ++i) {
      size_t thread_index = 0;  // omp_get_thread_num();
      int batch_size = static_cast<int>(
          std::min(num_rows - i * batch_size_max, static_cast<int64_t>(batch_size_max)));
      if (target->NumHashBitsUsed() > 32) {
        uint64_t* hashes = thread_local_hashes64[thread_index].data();
        get_hash64_impl(i * batch_size_max, batch_size, hashes);
        Status status = builder->PushNextBatch(thread_index, batch_size, hashes);
        ARROW_DCHECK(status.ok());
      } else {
        uint32_t* hashes = thread_local_hashes32[thread_index].data();
        get_hash32_impl(i * batch_size_max, batch_size, hashes);
        Status status = builder->PushNextBatch(thread_index, batch_size, hashes);
        ARROW_DCHECK(status.ok());
      }
    }

    // #pragma omp parallel for
    for (int64_t i = 0; i < builder->num_tasks(); ++i) {
      builder->RunFinishTask(i);
    }

    auto time1 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();

    builder->CleanUp();

    build_cost_vector[irepeat] = static_cast<float>(ns) / static_cast<float>(num_rows);
  }

  std::sort(build_cost_vector.begin(), build_cost_vector.end());
  *build_cost = build_cost_vector[build_cost_vector.size() / 2];

  return Status::OK();
}

// FPR (false positives rate) - fraction of false positives relative to the sum
// of false positives and true negatives.
//
// Output FPR and build and probe cost.
//
Status TestBloomSmall(BloomFilterBuildStrategy strategy, int64_t num_build,
                      int num_build_copies, int dop, bool use_simd, bool enable_prefetch,
                      float* fpr, float* build_cost, float* probe_cost) {
  int64_t hardware_flags = use_simd ? ::arrow::internal::CpuInfo::AVX2 : 0;

  // Generate input keys
  //
  int64_t num_probe = 4 * num_build;
  Random64Bit rnd(/*seed=*/0);
  std::vector<uint64_t> unique_keys;
  {
    std::set<uint64_t> unique_keys_set;
    for (int64_t i = 0; i < num_build + num_probe; ++i) {
      uint64_t value;
      for (;;) {
        value = rnd.next();
        if (unique_keys_set.find(value) == unique_keys_set.end()) {
          break;
        }
      }
      unique_keys.push_back(value);
      unique_keys_set.insert(value);
    }
  }

  // Generate input hashes
  //
  std::vector<uint32_t> hashes32;
  std::vector<uint64_t> hashes64;
  hashes32.resize(unique_keys.size());
  hashes64.resize(unique_keys.size());
  int batch_size_max = 1024;
  for (size_t i = 0; i < unique_keys.size(); i += batch_size_max) {
    int batch_size = static_cast<int>(
        std::min(unique_keys.size() - i, static_cast<size_t>(batch_size_max)));
    int key_length = sizeof(uint64_t);
    Hashing32::hash_fixed(hardware_flags, /*combine_hashes=*/false, batch_size,
                          key_length,
                          reinterpret_cast<const uint8_t*>(unique_keys.data() + i),
                          hashes32.data() + i, nullptr);
    Hashing64::hash_fixed(
        /*combine_hashes=*/false, batch_size, key_length,
        reinterpret_cast<const uint8_t*>(unique_keys.data() + i), hashes64.data() + i);
  }

  MemoryPool* pool = default_memory_pool();

  // Build the filter
  //
  BlockedBloomFilter reference;
  BlockedBloomFilter bloom;
  float build_cost_single_threaded;

  RETURN_NOT_OK(BuildBloomFilter(
      BloomFilterBuildStrategy::SINGLE_THREADED, dop, hardware_flags, pool, num_build,
      [hashes32](int64_t first_row, int num_rows, uint32_t* output_hashes) {
        memcpy(output_hashes, hashes32.data() + first_row, num_rows * sizeof(uint32_t));
      },
      [hashes64](int64_t first_row, int num_rows, uint64_t* output_hashes) {
        memcpy(output_hashes, hashes64.data() + first_row, num_rows * sizeof(uint64_t));
      },
      &reference, &build_cost_single_threaded));

  RETURN_NOT_OK(BuildBloomFilter(
      strategy, dop, hardware_flags, pool, num_build * num_build_copies,
      [hashes32, num_build](int64_t first_row, int num_rows, uint32_t* output_hashes) {
        int64_t first_row_clamped = first_row % num_build;
        int64_t num_rows_processed = 0;
        while (num_rows_processed < num_rows) {
          int64_t num_rows_next =
              std::min(static_cast<int64_t>(num_rows) - num_rows_processed,
                       num_build - first_row_clamped);
          memcpy(output_hashes + num_rows_processed, hashes32.data() + first_row_clamped,
                 num_rows_next * sizeof(uint32_t));
          first_row_clamped = 0;
          num_rows_processed += num_rows_next;
        }
      },
      [hashes64, num_build](int64_t first_row, int num_rows, uint64_t* output_hashes) {
        int64_t first_row_clamped = first_row % num_build;
        int64_t num_rows_processed = 0;
        while (num_rows_processed < num_rows) {
          int64_t num_rows_next =
              std::min(static_cast<int64_t>(num_rows) - num_rows_processed,
                       num_build - first_row_clamped);
          memcpy(output_hashes + num_rows_processed, hashes64.data() + first_row_clamped,
                 num_rows_next * sizeof(uint64_t));
          first_row_clamped = 0;
          num_rows_processed += num_rows_next;
        }
      },
      &bloom, build_cost));

  int log_before = bloom.log_num_blocks();

  if (num_build_copies > 1) {
    reference.Fold();
    bloom.Fold();
  } else {
    if (strategy != BloomFilterBuildStrategy::SINGLE_THREADED) {
      bool is_same = reference.IsSameAs(&bloom);
      printf("%s ", is_same ? "BUILD_CORRECT" : "BUILD_WRONG");
      ARROW_DCHECK(is_same);
    }
  }

  int log_after = bloom.log_num_blocks();

  float fraction_of_bits_set = static_cast<float>(bloom.NumBitsSet()) /
                               static_cast<float>(64LL << bloom.log_num_blocks());

  printf("log_before = %d log_after = %d percent_bits_set = %.1f ", log_before, log_after,
         100.0f * fraction_of_bits_set);

  // Verify no false negatives
  //
  bool ok = true;
  for (int64_t i = 0; i < num_build; ++i) {
    bool found;
    if (bloom.NumHashBitsUsed() > 32) {
      found = bloom.Find(hashes64[i]);
    } else {
      found = bloom.Find(hashes32[i]);
    }
    if (!found) {
      ok = false;
      ARROW_DCHECK(false);
      break;
    }
  }
  printf("%s\n", ok ? "success" : "failure");

  // Measure FPR and performance
  //
  std::vector<uint8_t> result_bit_vector;
  result_bit_vector.resize(bit_util::BytesForBits(batch_size_max));
  std::atomic<int64_t> num_positives;
  num_positives.store(0);

  int64_t num_repeats = 1LL;
#ifdef NDEBUG
  num_repeats = std::max(1LL, bit_util::CeilDiv(1000000ULL, num_probe));
#endif

  auto time0 = std::chrono::high_resolution_clock::now();

  for (int64_t irepeat = 0; irepeat < num_repeats; ++irepeat) {
    for (int64_t i = num_build; i < num_build + num_probe;) {
      int batch_size =
          static_cast<int>(std::min(static_cast<size_t>(unique_keys.size() - i),
                                    static_cast<size_t>(batch_size_max)));
      if (bloom.NumHashBitsUsed() > 32) {
        bloom.Find(hardware_flags, batch_size, hashes64.data() + i,
                   result_bit_vector.data(), enable_prefetch);
      } else {
        bloom.Find(hardware_flags, batch_size, hashes32.data() + i,
                   result_bit_vector.data(), enable_prefetch);
      }
      num_positives += arrow::internal::CountSetBits(result_bit_vector.data(),
                                                     /*offset=*/0, batch_size);
      i += batch_size;
    }
  }
  auto time1 = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
  *probe_cost = static_cast<float>(ns) / static_cast<float>(num_probe * num_repeats);

  *fpr = 100.0f * static_cast<float>(num_positives.load()) /
         static_cast<float>(num_probe * num_repeats);

  return Status::OK();
}

template <typename T>
void test_Bloom_large_hash(int64_t hardware_flags, int64_t block,
                           const std::vector<uint64_t>& first_in_block, int64_t first_row,
                           int num_rows, T* output_hashes) {
  // Largest 63-bit prime
  constexpr uint64_t prime = 0x7FFFFFFFFFFFFFE7ULL;

  constexpr int mini_batch_size = 1024;
  uint64_t keys[mini_batch_size];
  int64_t ikey = first_row / block * block;
  uint64_t key = first_in_block[first_row / block];
  while (ikey < first_row) {
    key += prime;
    ++ikey;
  }
  for (int ibase = 0; ibase < num_rows;) {
    int next_batch_size = std::min(num_rows - ibase, mini_batch_size);
    for (int i = 0; i < next_batch_size; ++i) {
      keys[i] = key;
      key += prime;
    }

    int key_length = sizeof(uint64_t);
    if (sizeof(T) == sizeof(uint32_t)) {
      Hashing32::hash_fixed(hardware_flags, false, next_batch_size, key_length,
                            reinterpret_cast<const uint8_t*>(keys),
                            reinterpret_cast<uint32_t*>(output_hashes) + ibase, nullptr);
    } else {
      Hashing64::hash_fixed(false, next_batch_size, key_length,
                            reinterpret_cast<const uint8_t*>(keys),
                            reinterpret_cast<uint64_t*>(output_hashes) + ibase);
    }

    ibase += next_batch_size;
  }
}

// Test with larger size Bloom filters (use large prime with arithmetic
// sequence modulo 2^64).
//
Status TestBloomLarge(BloomFilterBuildStrategy strategy, int64_t num_build, int dop,
                      bool use_simd, bool enable_prefetch, float* fpr, float* build_cost,
                      float* probe_cost) {
  int64_t hardware_flags = use_simd ? ::arrow::internal::CpuInfo::AVX2 : 0;

  // Largest 63-bit prime
  constexpr uint64_t prime = 0x7FFFFFFFFFFFFFE7ULL;

  // Generate input keys
  //
  int64_t num_probe = 4 * num_build;
  const int64_t block = 1024;
  std::vector<uint64_t> first_in_block;
  first_in_block.resize(bit_util::CeilDiv(num_build + num_probe, block));
  uint64_t current = prime;
  for (int64_t i = 0; i < num_build + num_probe; ++i) {
    if (i % block == 0) {
      first_in_block[i / block] = current;
    }
    current += prime;
  }

  MemoryPool* pool = default_memory_pool();

  // Build the filter
  //
  BlockedBloomFilter reference;
  BlockedBloomFilter bloom;
  float build_cost_single_threaded;

  for (int ibuild = 0; ibuild < 2; ++ibuild) {
    if (ibuild == 0 && strategy == BloomFilterBuildStrategy::SINGLE_THREADED) {
      continue;
    }
    RETURN_NOT_OK(BuildBloomFilter(
        ibuild == 0 ? BloomFilterBuildStrategy::SINGLE_THREADED : strategy,
        ibuild == 0 ? 1 : dop, hardware_flags, pool, num_build,
        [hardware_flags, &first_in_block](int64_t first_row, int num_rows,
                                          uint32_t* output_hashes) {
          const int64_t block = 1024;
          test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                                num_rows, output_hashes);
        },
        [hardware_flags, &first_in_block](int64_t first_row, int num_rows,
                                          uint64_t* output_hashes) {
          const int64_t block = 1024;
          test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                                num_rows, output_hashes);
        },
        ibuild == 0 ? &reference : &bloom,
        ibuild == 0 ? &build_cost_single_threaded : build_cost));
  }

  if (strategy != BloomFilterBuildStrategy::SINGLE_THREADED) {
    bool is_same = reference.IsSameAs(&bloom);
    printf("%s ", is_same ? "BUILD_CORRECT" : "BUILD_WRONG");
    ARROW_DCHECK(is_same);
  }

  std::vector<uint32_t> hashes32;
  std::vector<uint64_t> hashes64;
  std::vector<uint8_t> result_bit_vector;
  hashes32.resize(block);
  hashes64.resize(block);
  result_bit_vector.resize(bit_util::BytesForBits(block));

  int64_t num_repeats = 1LL;
#ifdef NDEBUG
  num_repeats = std::max(1LL, bit_util::CeilDiv(1000000ULL, num_probe));
#endif

  // Verify no false negatives and measure false positives.
  // Measure FPR and performance.
  //
  int64_t num_negatives_build = 0LL;
  int64_t num_negatives_probe = 0LL;
  auto time0 = std::chrono::high_resolution_clock::now();

  for (int64_t i = 0; i < num_build + num_probe * num_repeats;) {
    int64_t first_row = i < num_build ? i : num_build + ((i - num_build) % num_probe);
    int64_t last_row = i < num_build ? num_build : num_build + num_probe;
    int64_t next_batch_size = std::min(last_row - first_row, block);
    if (bloom.NumHashBitsUsed() > 32) {
      test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                            static_cast<int>(next_batch_size), hashes64.data());
      bloom.Find(hardware_flags, next_batch_size, hashes64.data(),
                 result_bit_vector.data(), enable_prefetch);
    } else {
      test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                            static_cast<int>(next_batch_size), hashes32.data());
      bloom.Find(hardware_flags, next_batch_size, hashes32.data(),
                 result_bit_vector.data(), enable_prefetch);
    }
    uint64_t num_negatives = 0ULL;
    for (int iword = 0; iword < next_batch_size / 64; ++iword) {
      uint64_t word = reinterpret_cast<const uint64_t*>(result_bit_vector.data())[iword];
      num_negatives += ARROW_POPCOUNT64(~word);
    }
    if (next_batch_size % 64 > 0) {
      uint64_t word = reinterpret_cast<const uint64_t*>(
          result_bit_vector.data())[next_batch_size / 64];
      uint64_t mask = (1ULL << (next_batch_size % 64)) - 1;
      word |= ~mask;
      num_negatives += ARROW_POPCOUNT64(~word);
    }
    if (i < num_build) {
      num_negatives_build += num_negatives;
    } else {
      num_negatives_probe += num_negatives;
    }
    i += next_batch_size;
  }
  auto time1 = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();

  ARROW_DCHECK(num_negatives_build == 0);
  printf("%s false_negatives %d\n", num_negatives_build == 0 ? "success" : "failure",
         static_cast<int>(num_negatives_build));

  *probe_cost = static_cast<float>(ns) / static_cast<float>(num_probe * num_repeats);

  *fpr = 100.0f * static_cast<float>(num_probe * num_repeats - num_negatives_probe) /
         static_cast<float>(num_probe * num_repeats);

  return Status::OK();
}

TEST(BloomFilter, Basic) {
  std::vector<int64_t> num_build;
  constexpr int log_min = 8;
  constexpr int log_max = 16;
  constexpr int log_large = 22;
  for (int log_num_build = log_min; log_num_build < log_max; ++log_num_build) {
    constexpr int num_intermediate_points = 2;
    for (int i = 0; i < num_intermediate_points; ++i) {
      int64_t num_left = 1LL << log_num_build;
      int64_t num_right = 1LL << (log_num_build + 1);
      num_build.push_back((num_left * (num_intermediate_points - i) + num_right * i) /
                          num_intermediate_points);
    }
  }
  num_build.push_back(1LL << log_max);
  num_build.push_back(1LL << log_large);

  constexpr int num_param_sets = 3;
  struct {
    bool use_avx2;
    bool enable_prefetch;
    bool insert_multiple_copies;
  } params[num_param_sets];
  for (int i = 0; i < num_param_sets; ++i) {
    params[i].use_avx2 = (i == 1);
    params[i].enable_prefetch = (i == 2);
    params[i].insert_multiple_copies = (i == 3);
  }

  std::vector<BloomFilterBuildStrategy> strategy;
  strategy.push_back(BloomFilterBuildStrategy::SINGLE_THREADED);
  strategy.push_back(BloomFilterBuildStrategy::PARALLEL);

  static constexpr int64_t min_rows_for_large = 2 * 1024 * 1024;

  int dop = 1;  // omp_get_max_threads();
  // printf("omp_get_thread_limit() = %d\n", dop);

  for (size_t istrategy = 0; istrategy < strategy.size(); ++istrategy) {
    for (int iparam_set = 0; iparam_set < num_param_sets; ++iparam_set) {
      printf("%s ", params[iparam_set].use_avx2                 ? "AVX2"
                    : params[iparam_set].enable_prefetch        ? "PREFETCH"
                    : params[iparam_set].insert_multiple_copies ? "FOLDING"
                                                                : "REGULAR");
      std::vector<float> fpr_vector(num_build.size());
      std::vector<float> probe_vector(num_build.size());
      for (size_t inum_build = 0; inum_build < num_build.size(); ++inum_build) {
        printf("num_build %d ", static_cast<int>(num_build[inum_build]));
        float fpr, build_cost, probe_cost;
        if (num_build[inum_build] >= min_rows_for_large) {
          ASSERT_OK(TestBloomLarge(strategy[istrategy], num_build[inum_build], dop,
                                   params[iparam_set].use_avx2,
                                   params[iparam_set].enable_prefetch, &fpr, &build_cost,
                                   &probe_cost));

        } else {
          ASSERT_OK(TestBloomSmall(strategy[istrategy], num_build[inum_build],
                                   params[iparam_set].insert_multiple_copies ? 8 : 1, dop,
                                   params[iparam_set].use_avx2,
                                   params[iparam_set].enable_prefetch, &fpr, &build_cost,
                                   &probe_cost));
        }
        if (iparam_set == 0) {
          fpr_vector[inum_build] = fpr;
        }
        probe_vector[inum_build] = probe_cost;
      }
      if (iparam_set == 0) {
        printf("(build size; FPR percent):\n");
        for (size_t i = 0; i < num_build.size(); ++i) {
          printf("%d; %.2f;\n", static_cast<int>(num_build[i]), fpr_vector[i]);
        }
      }
      printf("(build size; CPU cycles per probe):\n");
      for (size_t i = 0; i < num_build.size(); ++i) {
        printf("%d; %.2f;\n", static_cast<int>(num_build[i]), probe_vector[i]);
      }
    }
  }
}

TEST(BloomFilter, Scaling) {
  std::vector<int64_t> num_build;
  num_build.push_back(1000000);
  num_build.push_back(4000000);

  std::vector<int> dop;
  dop.push_back(1);

  std::vector<BloomFilterBuildStrategy> strategy;
  strategy.push_back(BloomFilterBuildStrategy::PARALLEL);

  for (bool use_avx2 : {false, true}) {
    for (size_t istrategy = 0; istrategy < strategy.size(); ++istrategy) {
      for (size_t inum_build = 0; inum_build < num_build.size(); ++inum_build) {
        std::vector<float> build_cost_per_row;
        build_cost_per_row.resize(dop.size());
        for (size_t idop = 0; idop < dop.size(); ++idop) {
          printf("num_build %d\n", static_cast<int>(num_build[inum_build]));
          printf("strategy %s ", strategy[istrategy] == BloomFilterBuildStrategy::PARALLEL
                                     ? "PARALLEL"
                                     : "SINGLE_THREADED");
          printf("%s ", use_avx2 ? "AVX2" : "SCALAR");
          printf("dop %d ", dop[idop]);
          float fpr = 100.0f, build_cost = 0.0f, probe_cost = 0.0f;
          ASSERT_OK(TestBloomLarge(
              strategy[istrategy], num_build[inum_build], dop[idop], use_avx2,
              /*enable_prefetch=*/false, &fpr, &build_cost, &probe_cost));
          build_cost_per_row[idop] = build_cost;
          printf("fpr %.2f build_cost %.1f probe_cost %.1f \n", fpr, build_cost,
                 probe_cost);
        }
        printf("(dop, build_cost_per_row):\n");
        for (size_t i = 0; i < build_cost_per_row.size(); ++i) {
          printf("%d; %.1f\n", dop[i], build_cost_per_row[i]);
        }
      }
    }
  }
}

}  // namespace compute
}  // namespace arrow
