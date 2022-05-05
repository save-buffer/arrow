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

#include <mutex>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/async_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"
#include "arrow/util/optional.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/tracing_internal.h"
#include "arrow/util/unreachable.h"
#include "arrow/util/vector.h"

namespace arrow {

using internal::checked_cast;
using internal::MapVector;

namespace compute {
namespace {

struct AsyncGenSourceNode : SourceNode {
  AsyncGenSourceNode(ExecPlan* plan, std::shared_ptr<Schema> output_schema,
             AsyncGenerator<util::optional<ExecBatch>> generator)
      : SourceNode(plan, std::move(output_schema)),
        generator_(std::move(generator)) {}

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 0, "AsyncGenSourceNode"));
    const auto& source_options = checked_cast<const SourceNodeOptions&>(options);
    return plan->EmplaceNode<AsyncGenSourceNode>(plan, source_options.output_schema,
                                                       source_options.generator);
  }

  const char* kind_name() const override { return "AsyncGenSourceNode"; }

  Status StartProducing() override {
    START_COMPUTE_SPAN(span_, std::string(kind_name()) + ":" + label(),
                       {{"node.kind", kind_name()},
                        {"node.label", label()},
                        {"node.output_schema", output_schema()->ToString()},
                        {"node.detail", ToString()}});

    auto executor = plan()->exec_context()->executor();
    ARROW_DCHECK(executor);
#if 0
    RETURN_NOT_OK(plan_->AddFuture(this->GenerateFuture()));
#endif
    return plan_->AddFuture(
        VisitAsyncGenerator(generator_,
                            [this](util::optional<ExecBatch> maybe_batch)
                            {
                                if(maybe_batch.has_value())
                                {
                                    size_t thread_index = plan()->GetThreadIndex();
                                    RETURN_NOT_OK(output_->InputReceived(thread_index, this, std::move(*maybe_batch)));
                                    batches_outputted_.fetch_add(1);
                                }
                                return Status::OK();
                            }).Then(
                                [this]()
                                {
                                    size_t thread_index = plan()->GetThreadIndex();
                                    return output_->InputFinished(thread_index, this, static_cast<int>(batches_outputted_.load()));
                                }));
    return Status::OK();
  }

  void PauseProducing(int32_t counter) override {
    std::lock_guard<std::mutex> lg(mutex_);
    if (counter <= backpressure_counter_) {
      return;
    }
    backpressure_counter_ = counter;
    if (!backpressure_future_.is_finished()) {
      // Could happen if we get something like Pause(1) Pause(3) Resume(2)
      return;
    }
    backpressure_future_ = Future<>::Make();
  }

  void ResumeProducing(int32_t counter) override {
    Future<> to_finish;
    {
      std::lock_guard<std::mutex> lg(mutex_);
      if (counter <= backpressure_counter_) {
        return;
      }
      backpressure_counter_ = counter;
      if (backpressure_future_.is_finished()) {
        return;
      }
      to_finish = backpressure_future_;
    }
    to_finish.MarkFinished();
  }

  Future<> finished() override { return finished_; }

 private:
#if 0
    Future<> GenerateFuture()
    {
        CallbackOptions options;
        options.executor = plan()->exec_context()->executor();
        options.should_schedule = ShouldSchedule::IfDifferentExecutor;
        return generator_().Then(
            [this](const util::optional<ExecBatch> &maybe_batch)
            {
                if(done_.load())
                {
                    ARROW_DCHECK(!maybe_batch.has_value()) << "Got a valid batch after supposedly finishing";
                    return Status::OK();
                }

                size_t thread_index = plan()->GetThreadIndex();
                if(maybe_batch.has_value())
                {
                    threads_outputting_.fetch_add(1);
                    RETURN_NOT_OK(output_->InputReceived(thread_index, this, std::move(*maybe_batch)));
                    batches_outputted_.fetch_add(1);
                    threads_outputting_.fetch_sub(1);
                    RETURN_NOT_OK(plan_->AddFuture(this->GenerateFuture()));
                }
                else
                {
                    if(threads_outputting_.load() == 0)
                    {
                        bool expected = false;
                        if(done_.compare_exchange_strong(expected, true))
                        {
                            RETURN_NOT_OK(output_->InputFinished(thread_index, this, static_cast<int>(batches_outputted_.load())));
                            END_SPAN(span_);
                        }
                    }
                }
                return Status::OK();
            }, {}, options);
    }
  std::atomic<bool> done_;
  std::atomic<int64_t> threads_outputting_{0};
#endif

  std::atomic<int64_t> batches_outputted_{0};
  std::mutex mutex_;
  int32_t backpressure_counter_{0};
  Future<> backpressure_future_ = Future<>::MakeFinished();
  AsyncGenerator<util::optional<ExecBatch>> generator_;
};

struct TableSourceNode : public AsyncGenSourceNode {
  TableSourceNode(ExecPlan* plan, std::shared_ptr<Table> table, int64_t batch_size)
      : AsyncGenSourceNode(plan, table->schema(), TableGenerator(*table, batch_size)) {}

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 0, "TableSourceNode"));
    const auto& table_options = checked_cast<const TableSourceNodeOptions&>(options);
    const auto& table = table_options.table;
    const int64_t batch_size = table_options.max_batch_size;

    RETURN_NOT_OK(ValidateTableSourceNodeInput(table, batch_size));

    return plan->EmplaceNode<TableSourceNode>(plan, table, batch_size);
  }

  const char* kind_name() const override { return "TableSourceNode"; }

  static arrow::Status ValidateTableSourceNodeInput(const std::shared_ptr<Table> table,
                                                    const int64_t batch_size) {
    if (table == nullptr) {
      return Status::Invalid("TableSourceNode node requires table which is not null");
    }

    if (batch_size <= 0) {
      return Status::Invalid(
          "TableSourceNode node requires, batch_size > 0 , but got batch size ",
          batch_size);
    }

    return Status::OK();
  }

  static arrow::AsyncGenerator<util::optional<ExecBatch>> TableGenerator(
      const Table& table, const int64_t batch_size) {
    auto batches = ConvertTableToExecBatches(table, batch_size);
    auto opt_batches =
        MapVector([](ExecBatch batch) { return util::make_optional(std::move(batch)); },
                  std::move(batches));
    AsyncGenerator<util::optional<ExecBatch>> gen;
    gen = MakeVectorGenerator(std::move(opt_batches));
    return gen;
  }

  static std::vector<ExecBatch> ConvertTableToExecBatches(const Table& table,
                                                          const int64_t batch_size) {
    std::shared_ptr<TableBatchReader> reader = std::make_shared<TableBatchReader>(table);

    // setting chunksize for the batch reader
    reader->set_chunksize(batch_size);

    std::shared_ptr<RecordBatch> batch;
    std::vector<ExecBatch> exec_batches;
    while (true) {
      auto batch_res = reader->Next();
      if (batch_res.ok()) {
        batch = std::move(batch_res).MoveValueUnsafe();
      }
      if (batch == NULLPTR) {
        break;
      }
      exec_batches.emplace_back(*batch);
    }
    return exec_batches;
  }
};

}  // namespace

namespace internal {

void RegisterSourceNode(ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("asyncgen_source", AsyncGenSourceNode::Make));
  DCHECK_OK(registry->AddFactory("table_source", TableSourceNode::Make));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
