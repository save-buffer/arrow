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

#include <bitset>

#include "arrow/compute/exec/query_context.h"
#include "arrow/compute/exec/hash_join.h"
#include "arrow/compute/exec/accumulation_queue.h"

namespace arrow
{
    namespace compute
    {
        struct PartitionedBloomFilter
        {
            std::unique_ptr<BlockedBloomFilter> in_memory;
            std::unique_ptr<BlockedBloomFilter> partitions[SpillingAccumulationQueue::kNumPartitions];

            void Find(
                int64_t hardware_flags,
                int64_t num_rows,
                const uint64_t *hashes,
                uint8_t *bv);
        };

        class SpillingHashJoin
        {
        public:
            using RegisterTaskGroupCallback = std::function<int(
                std::function<Status(size_t, int64_t)>, std::function<Status(size_t)>)>;
            using StartTaskGroupCallback = std::function<Status(int, int64_t)>;
            using AddProbeSideHashColumn = std::function<Status(size_t, ExecBatch *)>;
            using BloomFilterFinishedCallback = std::function<Status(size_t)>;
            using ApplyBloomFilterCallback = std::function<Status(size_t, ExecBatch *)>;
            using OutputBatchCallback = std::function<void(int64_t, ExecBatch)>;
            using FinishedCallback = std::function<Status(int64_t)>;
            using StartSpillingCallback = std::function<Status(size_t)>;
            using PauseProbeSideCallback = std::function<void(int)>;
            using ResumeProbeSideCallback = std::function<void(int)>;

            struct CallbackRecord
            {
                RegisterTaskGroupCallback register_task_group;
                StartTaskGroupCallback start_task_group;
                AddProbeSideHashColumn add_probe_side_hashes;
                BloomFilterFinishedCallback bloom_filter_finished;
                ApplyBloomFilterCallback apply_bloom_filter;
                OutputBatchCallback output_batch;
                FinishedCallback finished;
                StartSpillingCallback start_spilling;
                PauseProbeSideCallback pause_probe_side;
                ResumeProbeSideCallback resume_probe_side;
            };

            Status Init(
                QueryContext *ctx,
                JoinType join_type,
                size_t num_threads,
                SchemaProjectionMaps<HashJoinProjection> *proj_map_left,
                SchemaProjectionMaps<HashJoinProjection> *proj_map_right,
                std::vector<JoinKeyCmp> *key_cmp,
                Expression *filter,
                PartitionedBloomFilter *bloom_filter,
                CallbackRecord callback_record,
                bool is_swiss);

            Status CheckSpilling(size_t thread_index, ExecBatch &batch);

            Status OnBuildSideBatch(size_t thread_index, ExecBatch batch);
            Status OnBuildSideFinished(size_t thread_index);

            Status OnProbeSideBatch(size_t thread_index, ExecBatch batch);
            Status OnProbeSideFinished(size_t thread_index);

            Status OnBloomFiltersReceived(size_t thread_index);

        private:
            Status AdvanceSpillCursor(size_t thread_index);

            // Builds the entire bloom filter for all 64 partitions.
            Status BuildPartitionedBloomFilter(size_t thread_index);
            Status PushBloomFilterBatch(size_t thread_index, int64_t batch_id);
            // Builds a bloom filter for a single partition.
            Status BuildNextBloomFilter(size_t thread_index);
            Status OnBloomFilterFinished(size_t thread_index);
            Status OnPartitionedBloomFilterFinished(size_t thread_index);

            Status StartCollocatedJoins(size_t thread_index);
            Status BeginNextCollocatedJoin(size_t thread_index);
            Status BuildHashTable(size_t thread_index);
            Status OnHashTableFinished(size_t thread_index);
            Status OnProbeSideBatchReadBack(size_t thread_index, size_t batch_idx, ExecBatch batch);
            Status OnProbingFinished(size_t thread_index);
            Status OnCollocatedJoinFinished(int64_t num_batches);

            QueryContext *ctx_;
            size_t num_threads_;
            CallbackRecord callbacks_;
            bool is_swiss_;
            PartitionedBloomFilter *bloom_filter_;
            std::unique_ptr<BloomFilterBuilder> builder_;

            // Backpressure toggling happens at most twice during execution. A value of 0 means
            // we haven't toggled backpressure at all, value of 1 means we've paused, and value
            // 2 means we've resumed.
            std::atomic<int32_t> backpressure_counter_{0};

            SpillingAccumulationQueue build_accumulator_;
            SpillingAccumulationQueue probe_accumulator_;

            AccumulationQueue build_queue_;

            std::unique_ptr<HashJoinImpl> impls_[SpillingAccumulationQueue::kNumPartitions];
            int task_group_bloom_[SpillingAccumulationQueue::kNumPartitions];

            std::atomic<size_t> max_batch_size_{0};

            int64_t total_batches_outputted_ = 0;
            size_t partition_idx_ = SpillingAccumulationQueue::kNumPartitions;
            std::atomic<bool> spilling_{false};
            std::atomic<bool> bloom_or_probe_finished_{false};
            std::atomic<bool> bloom_ready_{false};
        };
    }
}
