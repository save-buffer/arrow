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

#include "arrow/compute/exec/exec_plan.h"

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "arrow/util/checked_cast.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/datum.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/logging.h"
#include "arrow/util/optional.h"
#include "arrow/util/tracing_internal.h"
#include "arrow/util/unreachable.h"

namespace arrow {

using internal::checked_cast;

namespace compute {

namespace {

using arrow::internal::checked_cast;

struct ExecPlanImpl : public ExecPlan {
  explicit ExecPlanImpl(ExecContext* exec_context,
                        std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR)
      : ExecPlan(exec_context), metadata_(std::move(metadata)) {}

  ~ExecPlanImpl() override {
    if (started_ && !finished_.is_finished()) {
      ARROW_LOG(WARNING) << "Plan was destroyed before finishing";
      Abort();
      finished().Wait();
    }
  }

  ExecNode* AddNode(std::unique_ptr<ExecNode> node) {
    if (node->label().empty()) {
      node->SetLabel(std::to_string(auto_label_counter_++));
    }
    nodes_.push_back(std::move(node));
    return nodes_.back().get();
  }

    ExecNode* AddSourceNode(std::unique_ptr<ExecNode> node)
    {
        sources_.push_back(checked_cast<SourceNode *>(node.get()));
        return AddNode(std::move(node));
    }

    ExecNode* AddSinkNode(std::unique_ptr<ExecNode> node)
    {
        sinks_.push_back(checked_cast<SinkNode *>(node.get()));
        return AddNode(std::move(node));
    }

  Status Validate() const {
    if (nodes_.empty())
      return Status::Invalid("ExecPlan has no node");
    if(sources_.empty())
        return Status::Invalid("ExecPlan has no sources");
    if(sinks_.empty())
        return Status::Invalid("ExecPlan has no sinks");
    for (const auto& node : nodes_) {
      RETURN_NOT_OK(node->Validate());
    }
    return Status::OK();
  }

  Status StartProducing() {
    START_COMPUTE_SPAN(span_, "ExecPlan", {{"plan", ToString()}});
#ifdef ARROW_WITH_OPENTELEMETRY
    if (HasMetadata()) {
      auto pairs = metadata().get()->sorted_pairs();
      std::for_each(std::begin(pairs), std::end(pairs),
                    [this](std::pair<std::string, std::string> const& pair) {
                      span_.Get().span->SetAttribute(pair.first, pair.second);
                    });
    }
#endif
    if (started_) {
      return Status::Invalid("restarted ExecPlan");
    }

    for(const auto &n : nodes_)
        RETURN_NOT_OK(n->Init());

    std::vector<Future<>> futures;
    for(ExecNode *n : sinks())
        futures.push_back(n->finished());

    // When all of the sink nodes are finished, end the task group.
    AllFinished(futures).AddCallback([this](const Status &)
    {
        if(!aborted_)
            std::ignore = task_group_.End();
    });
    // When the task group is finished, mark the plan finished. If we aborted,
    // perform any cleanup.
    task_group_.OnFinished().AddCallback(
        [this](const Status &)
        {
            Status st = Status::OK();
            if(aborted_)
            {
                for(const auto &n : nodes_)
                    n->Abort();
                if(!errors_.empty())
                    st = std::move(errors_.front());
            }
            finished_.MarkFinished(std::move(st));
        });


    started_ = true;
    Status st = Status::OK();
    for(SourceNode *n : sources())
    {
        EVENT(span_, "StartProducing:" + node->label(),
              {{"node.label", node->label()}, {"node.kind_name", node->kind_name()}});
        st = n->StartProducing();
        EVENT(span_, "StartProducing:" + node->label(), {{"status", st.ToString()}});
        if(!st.ok())
        {
            stopped_ = true;
            Abort();
            return st;
        }
    }
    END_SPAN_ON_FUTURE_COMPLETION(span_, finished_, this);
    return Status::OK();
  }

  void Abort() {
      if(finished_.is_finished())
          return;
      std::lock_guard<std::mutex> guard(abort_mutex_);
      AbortUnlocked();
  }

    void AbortUnlocked()
    {
        if(!aborted_)
        {
            aborted_ = true;
            DCHECK(started_) << "aborted an ExecPlan which never started";
            EVENT(span_, "Abort");
            
            stop_source_.RequestStop();
            std::ignore = task_group_.End();
        }
    }

    size_t num_threads() const { return thread_indexer_.Capacity(); }
    size_t GetThreadIndex() { return thread_indexer_(); }

    Status AddFuture(Future<> fut)
    {
        fut.AddCallback([this](const Status &status)
        {
            if(!status.ok())
            {
                std::lock_guard<std::mutex> guard(abort_mutex_);
                errors_.emplace_back(std::move(status));
                AbortUnlocked();
            }
        });
        return task_group_.AddTaskIfNotEnded(std::move(fut));
    }

    Status ScheduleTask(const std::atomic<bool> &is_paused, std::function<Status(size_t)> fn)
    {
        auto executor = exec_context_->executor();
        ARROW_DCHECK(executor);
        ARROW_ASSIGN_OR_RAISE(auto task_fut, executor->Submit(
                                  stop_source_.token(), &is_paused, [this, fn]()
                                  {
                                      size_t thread_index = thread_indexer_();
                                      return fn(thread_index);
                                  }));
        return task_group_.AddTaskIfNotEnded(std::move(task_fut));
    }

  NodeVector TopoSort() const {
    struct Impl {
      const std::vector<std::unique_ptr<ExecNode>>& nodes;
      std::unordered_set<ExecNode*> visited;
      NodeVector sorted;

      explicit Impl(const std::vector<std::unique_ptr<ExecNode>>& nodes) : nodes(nodes) {
        visited.reserve(nodes.size());
        sorted.resize(nodes.size());

        for (const auto& node : nodes) {
          Visit(node.get());
        }

        DCHECK_EQ(visited.size(), nodes.size());
      }

      void Visit(ExecNode* node) {
        if (visited.count(node) != 0) return;

        for (auto input : node->inputs()) {
          // Ensure that producers are inserted before this consumer
          Visit(input);
        }

        sorted[visited.size()] = node;
        visited.insert(node);
      }
    };

    return std::move(Impl{nodes_}.sorted);
  }

  // This function returns a node vector and a vector of integers with the
  // number of spaces to add as an indentation. The main difference between
  // this function and the TopoSort function is that here we visit the nodes
  // in reverse order and we can have repeated nodes if necessary.
  // For example, in the following plan:
  // s1 --> s3 -
  //   -        -
  //    -        -> s5 --> s6
  //     -      -
  // s2 --> s4 -
  // Toposort node vector: s1 s2 s3 s4 s5 s6
  // OrderedNodes node vector: s6 s5 s3 s1 s4 s2 s1
  std::pair<NodeVector, std::vector<int>> OrderedNodes() const {
    struct Impl {
      const std::vector<std::unique_ptr<ExecNode>>& nodes;
      std::unordered_set<ExecNode*> visited;
      std::unordered_set<ExecNode*> marked;
      NodeVector sorted;
      std::vector<int> indents;

      explicit Impl(const std::vector<std::unique_ptr<ExecNode>>& nodes) : nodes(nodes) {
        visited.reserve(nodes.size());

        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
          if (visited.count(it->get()) != 0) continue;
          Visit(it->get());
        }

        DCHECK_EQ(visited.size(), nodes.size());
      }

      void Visit(ExecNode* node, int indent = 0) {
        marked.insert(node);
        for (auto input : node->inputs()) {
          if (marked.count(input) != 0) continue;
          Visit(input, indent + 1);
        }
        marked.erase(node);

        indents.push_back(indent);
        sorted.push_back(node);
        visited.insert(node);
      }
    };

    auto result = Impl{nodes_};
    return std::make_pair(result.sorted, result.indents);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "ExecPlan with " << nodes_.size() << " nodes:" << std::endl;
    auto sorted = OrderedNodes();
    for (size_t i = sorted.first.size(); i > 0; --i) {
      for (int j = 0; j < sorted.second[i - 1]; ++j) ss << "  ";
      ss << sorted.first[i - 1]->ToString(sorted.second[i - 1]) << std::endl;
    }
    return ss.str();
  }

  Future<> finished_ = Future<>::Make();
  bool started_ = false, stopped_ = false;
  std::vector<std::unique_ptr<ExecNode>> nodes_;
  util::AsyncTaskGroup task_group_;
  ThreadIndexer thread_indexer_;
  std::vector<SourceNode *> sources_;
  std::vector<SinkNode *> sinks_;
  NodeVector sorted_nodes_;
  uint32_t auto_label_counter_ = 0;
  util::tracing::Span span_;
  std::shared_ptr<const KeyValueMetadata> metadata_;

  std::mutex abort_mutex_;
  bool aborted_ = false;
  StopSource stop_source_;
  std::vector<Status> errors_;
};

ExecPlanImpl* ToDerived(ExecPlan* ptr) { return checked_cast<ExecPlanImpl*>(ptr); }

const ExecPlanImpl* ToDerived(const ExecPlan* ptr) {
  return checked_cast<const ExecPlanImpl*>(ptr);
}

}  // namespace

Result<std::shared_ptr<ExecPlan>> ExecPlan::Make(
    ExecContext* ctx, std::shared_ptr<const KeyValueMetadata> metadata) {
  return std::shared_ptr<ExecPlan>(new ExecPlanImpl{ctx, metadata});
}

ExecNode* ExecPlan::AddSourceNode(std::unique_ptr<ExecNode> node) {
  return ToDerived(this)->AddSourceNode(std::move(node));
}

ExecNode* ExecPlan::AddNode(std::unique_ptr<ExecNode> node) {
  return ToDerived(this)->AddNode(std::move(node));
}

ExecNode* ExecPlan::AddSinkNode(std::unique_ptr<ExecNode> node) {
  return ToDerived(this)->AddSinkNode(std::move(node));
}


const std::vector<SourceNode *>& ExecPlan::sources() const {
  return ToDerived(this)->sources_;
}

const std::vector<SinkNode *>& ExecPlan::sinks() const { return ToDerived(this)->sinks_; }

Status ExecPlan::Validate() { return ToDerived(this)->Validate(); }

Status ExecPlan::StartProducing() { return ToDerived(this)->StartProducing(); }

void ExecPlan::Abort() { ToDerived(this)->Abort(); }

size_t ExecPlan::num_threads() const { return ToDerived(this)->num_threads(); }
size_t ExecPlan::GetThreadIndex() { return ToDerived(this)->GetThreadIndex(); }
Status ExecPlan::AddFuture(Future<> fut) { return ToDerived(this)->AddFuture(std::move(fut)); }
Status ExecPlan::ScheduleTask(const std::atomic<bool> &pause_toggle, std::function<Status(size_t)> fn) { return ToDerived(this)->ScheduleTask(pause_toggle, std::move(fn)); }

Future<> ExecPlan::finished() { return ToDerived(this)->finished_; }

bool ExecPlan::HasMetadata() const { return !!(ToDerived(this)->metadata_); }

std::shared_ptr<const KeyValueMetadata> ExecPlan::metadata() const {
  return ToDerived(this)->metadata_;
}

std::string ExecPlan::ToString() const { return ToDerived(this)->ToString(); }

ExecNode::ExecNode(ExecPlan* plan, NodeVector inputs,
                   std::vector<std::string> input_labels,
                   std::shared_ptr<Schema> output_schema)
    : plan_(plan),
      inputs_(std::move(inputs)),
      input_labels_(std::move(input_labels)),
      output_schema_(std::move(output_schema)),
      output_(nullptr) {
    for(auto input : inputs_)
    {
        // We allow duplicates of the inputs in the input list
        ARROW_DCHECK(input->output_ == nullptr || input->output_ == this);
        input->output_ = this;
    }
}

Status ExecNode::Validate() const {
  if (inputs_.size() != input_labels_.size()) {
    return Status::Invalid("Invalid number of inputs for '", label(), "' (expected ",
                           num_inputs(), ", actual ", input_labels_.size(), ")");
  }

  for(auto input : inputs_)
  {
      if(input->output_ != this)
          return Status::Invalid("Node '", input->label(), "' is listed as an input to '", label(),
                                 "', but '", input->label(), "''s output_ field is not set properly.");

  }
  return Status::OK();
}

std::string ExecNode::ToString(int indent) const {
  std::stringstream ss;

  auto PrintLabelAndKind = [&](const ExecNode* node) {
    ss << node->label() << ":" << node->kind_name();
  };

  PrintLabelAndKind(this);
  ss << "{";

  const std::string extra = ToStringExtra(indent);
  if (!extra.empty()) {
    ss << extra;
  }

  ss << '}';
  return ss.str();
}

std::string ExecNode::ToStringExtra(int indent = 0) const { return ""; }

bool ExecNode::ErrorIfNotOk(Status status) {
  if (status.ok()) return false;
  return true;
}

    Status SourceNode::Validate() const
    {
        ARROW_DCHECK(inputs_.empty()) << "You get a gold star for managing to construct a SourceNode with non-empty inputs.";
        if(output_ == nullptr)
            return Status::Invalid("Source node '", label(), "' has no output.");
        return ExecNode::Validate();
    }

void SourceNode::NoInputs() {
    Unreachable("no inputs; this should never be called");
}

    Status SinkNode::Validate() const
    {
        ARROW_DCHECK(output_ == nullptr) << "You get a gold star for managing to construct a SinkNode with an output.";
        if(inputs_.empty())
            return Status::Invalid("Sink node '", label(), "' has no inputs.");
        return ExecNode::Validate();
    }

void SinkNode::NoOutputs() {
    Unreachable("no outputs; this should never be called");
}

MapNode::MapNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
                 std::shared_ptr<Schema> output_schema)
    : ExecNode(plan, std::move(inputs), /*input_labels=*/{"target"},
               std::move(output_schema)) {
}

Status MapNode::InputFinished(size_t thread_index, ExecNode* input, int total_batches) {
  DCHECK_EQ(input, inputs_[0]);
  EVENT(span_, "InputFinished", {{"batches.length", total_batches}});
  RETURN_NOT_OK(output_->InputFinished(thread_index, this, total_batches));
  if (input_counter_.SetTotal(total_batches)) {
    this->Finish();
  }
  return Status::OK();
}

void MapNode::PauseProducing(int32_t counter) {
  inputs_[0]->PauseProducing(counter);
}

void MapNode::ResumeProducing(int32_t counter) {
  inputs_[0]->ResumeProducing(counter);
}

void MapNode::Abort() {
  EVENT(span_, "Abort");
  if (executor_) {
    this->stop_source_.RequestStop();
  }
  if (input_counter_.Cancel()) {
    this->Finish();
  }
}

Future<> MapNode::finished() { return finished_; }

Status MapNode::DoMap(size_t thread_index,
                      std::function<Result<ExecBatch>(ExecBatch)> map_fn,
                      ExecBatch batch) {
  // This will be true if the node is stopped early due to an error or manual
  // cancellation
  if (input_counter_.Completed()) {
      return Status::OK();
  }
  auto guarantee = batch.guarantee;
  ARROW_ASSIGN_OR_RAISE(auto output_batch, map_fn(std::move(batch)));
  output_batch.guarantee = guarantee;
  RETURN_NOT_OK(output_->InputReceived(thread_index, this, std::move(output_batch)));
  if (input_counter_.Increment()) {
    this->Finish();
  }
  return Status::OK();
}

void MapNode::Finish(Status finish_st /*= Status::OK()*/) {
  this->finished_.MarkFinished(finish_st);
}

std::shared_ptr<RecordBatchReader> MakeGeneratorReader(
    std::shared_ptr<Schema> schema,
    std::function<Future<util::optional<ExecBatch>>()> gen, MemoryPool* pool) {
  struct Impl : RecordBatchReader {
    std::shared_ptr<Schema> schema() const override { return schema_; }

    Status ReadNext(std::shared_ptr<RecordBatch>* record_batch) override {
      ARROW_ASSIGN_OR_RAISE(auto batch, iterator_.Next());
      if (batch) {
        ARROW_ASSIGN_OR_RAISE(*record_batch, batch->ToRecordBatch(schema_, pool_));
      } else {
        *record_batch = IterationEnd<std::shared_ptr<RecordBatch>>();
      }
      return Status::OK();
    }

    MemoryPool* pool_;
    std::shared_ptr<Schema> schema_;
    Iterator<util::optional<ExecBatch>> iterator_;
  };

  auto out = std::make_shared<Impl>();
  out->pool_ = pool;
  out->schema_ = std::move(schema);
  out->iterator_ = MakeGeneratorIterator(std::move(gen));
  return out;
}

Result<ExecNode*> Declaration::AddToPlan(ExecPlan* plan,
                                         ExecFactoryRegistry* registry) const {
  std::vector<ExecNode*> inputs(this->inputs.size());

  size_t i = 0;
  for (const Input& input : this->inputs) {
    if (auto node = util::get_if<ExecNode*>(&input)) {
      inputs[i++] = *node;
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(inputs[i++],
                          util::get<Declaration>(input).AddToPlan(plan, registry));
  }

  ARROW_ASSIGN_OR_RAISE(
      auto node, MakeExecNode(this->factory_name, plan, std::move(inputs), *this->options,
                              registry));
  node->SetLabel(this->label);
  return node;
}

Declaration Declaration::Sequence(std::vector<Declaration> decls) {
  DCHECK(!decls.empty());

  Declaration out = std::move(decls.back());
  decls.pop_back();
  auto receiver = &out;
  while (!decls.empty()) {
    Declaration input = std::move(decls.back());
    decls.pop_back();

    receiver->inputs.emplace_back(std::move(input));
    receiver = &util::get<Declaration>(receiver->inputs.front());
  }
  return out;
}

namespace internal {

void RegisterSourceNode(ExecFactoryRegistry*);
void RegisterFilterNode(ExecFactoryRegistry*);
void RegisterProjectNode(ExecFactoryRegistry*);
void RegisterUnionNode(ExecFactoryRegistry*);
void RegisterAggregateNode(ExecFactoryRegistry*);
void RegisterSinkNode(ExecFactoryRegistry*);
void RegisterHashJoinNode(ExecFactoryRegistry*);

}  // namespace internal

ExecFactoryRegistry* default_exec_factory_registry() {
  class DefaultRegistry : public ExecFactoryRegistry {
   public:
    DefaultRegistry() {
      internal::RegisterSourceNode(this);
      internal::RegisterFilterNode(this);
      internal::RegisterProjectNode(this);
      internal::RegisterUnionNode(this);
      internal::RegisterAggregateNode(this);
      internal::RegisterSinkNode(this);
      internal::RegisterHashJoinNode(this);
    }

    Result<Factory> GetFactory(const std::string& factory_name) override {
      auto it = factories_.find(factory_name);
      if (it == factories_.end()) {
        return Status::KeyError("ExecNode factory named ", factory_name,
                                " not present in registry.");
      }
      return it->second;
    }

    Status AddFactory(std::string factory_name, Factory factory) override {
      auto it_success = factories_.emplace(std::move(factory_name), std::move(factory));

      if (!it_success.second) {
        const auto& factory_name = it_success.first->first;
        return Status::KeyError("ExecNode factory named ", factory_name,
                                " already registered.");
      }

      return Status::OK();
    }

   private:
    std::unordered_map<std::string, Factory> factories_;
  };

  static DefaultRegistry instance;
  return &instance;
}

Result<std::function<Future<util::optional<ExecBatch>>()>> MakeReaderGenerator(
    std::shared_ptr<RecordBatchReader> reader, ::arrow::internal::Executor* io_executor,
    int max_q, int q_restart) {
  auto batch_it = MakeMapIterator(
      [](std::shared_ptr<RecordBatch> batch) {
        return util::make_optional(ExecBatch(*batch));
      },
      MakeIteratorFromReader(reader));

  return MakeBackgroundGenerator(std::move(batch_it), io_executor, max_q, q_restart);
}

}  // namespace compute
}  // namespace arrow
