#include "graph/StrictSync.h"

#include <algorithm>
#include <chrono>

namespace simaai::neat::graph::strict_sync {

PendingVideoStore::PendingVideoStore(size_t streams) : states_(streams) {}

bool PendingVideoStore::enqueue(size_t idx, int64_t frame_id, simaai::neat::Sample&& sample,
                                int64_t cap_ms, size_t bytes) {
  if (idx >= states_.size() || frame_id < 0)
    return false;
  auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);

  auto dup = st.pending.find(frame_id);
  if (dup != st.pending.end()) {
    st.bytes_total =
        (dup->second.bytes <= st.bytes_total) ? (st.bytes_total - dup->second.bytes) : 0;
    st.pending.erase(dup);
    auto oit = std::find(st.order.begin(), st.order.end(), frame_id);
    if (oit != st.order.end())
      st.order.erase(oit);
  }

  PendingFrame pf;
  pf.sample = std::move(sample);
  pf.cap_ms = cap_ms;
  pf.bytes = bytes;

  st.order.push_back(frame_id);
  st.pending.emplace(frame_id, std::move(pf));
  st.bytes_total += bytes;

  st.stats.enqueued++;
  st.stats.pending_depth = st.pending.size();
  st.stats.pending_bytes = st.bytes_total;
  st.stats.max_pending_depth = std::max(st.stats.max_pending_depth, st.stats.pending_depth);
  st.stats.max_pending_bytes = std::max(st.stats.max_pending_bytes, st.stats.pending_bytes);
  return true;
}

std::optional<PendingVideoStore::PendingFrame> PendingVideoStore::take(size_t idx,
                                                                       int64_t frame_id) {
  if (idx >= states_.size() || frame_id < 0)
    return std::nullopt;
  auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);

  auto it = st.pending.find(frame_id);
  if (it == st.pending.end()) {
    st.stats.miss++;
    st.stats.pending_depth = st.pending.size();
    st.stats.pending_bytes = st.bytes_total;
    return std::nullopt;
  }

  PendingFrame out = std::move(it->second);
  st.bytes_total = (out.bytes <= st.bytes_total) ? (st.bytes_total - out.bytes) : 0;
  st.pending.erase(it);

  auto oit = std::find(st.order.begin(), st.order.end(), frame_id);
  if (oit != st.order.end())
    st.order.erase(oit);

  st.stats.matched++;
  st.stats.pending_depth = st.pending.size();
  st.stats.pending_bytes = st.bytes_total;
  return out;
}

PendingVideoStore::StreamStats PendingVideoStore::stats(size_t idx) const {
  if (idx >= states_.size())
    return StreamStats{};
  const auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);
  return st.stats;
}

YoloTokenStore::YoloTokenStore(size_t streams) : states_(streams) {}

int64_t YoloTokenStore::now_ms_i64() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

void YoloTokenStore::enqueue(size_t idx, int64_t frame_id) {
  if (idx >= states_.size() || frame_id < 0)
    return;

  const int64_t now_ms = now_ms_i64();
  {
    std::lock_guard<std::mutex> lk(order_mu_);
    order_q_.push_back(OrderedToken{idx, frame_id, now_ms});
  }

  auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);
  st.q.push_back(Token{frame_id, now_ms});
  st.stats.enqueued++;
  st.stats.depth = st.q.size();
  st.stats.max_depth = std::max(st.stats.max_depth, st.stats.depth);
}

std::optional<YoloTokenStore::OrderedToken> YoloTokenStore::take_ordered() {
  std::lock_guard<std::mutex> lk(order_mu_);
  if (order_q_.empty())
    return std::nullopt;
  OrderedToken t = order_q_.front();
  order_q_.pop_front();
  return t;
}

std::optional<YoloTokenStore::Token> YoloTokenStore::take(size_t idx) {
  if (idx >= states_.size())
    return std::nullopt;
  auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);
  if (st.q.empty()) {
    st.stats.miss++;
    st.stats.depth = 0;
    return std::nullopt;
  }
  Token t = st.q.front();
  st.q.pop_front();
  st.stats.dequeued++;
  st.stats.depth = st.q.size();
  return t;
}

YoloTokenStore::Stats YoloTokenStore::stats(size_t idx) const {
  if (idx >= states_.size())
    return Stats{};
  const auto& st = states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);
  return st.stats;
}

ReleasePacer::ReleasePacer(const std::vector<std::shared_ptr<simaai::neat::Run>>& runs,
                           int target_fps, size_t max_queue, OnSendResult on_send_result,
                           OnDrop on_drop)
    : runs_(runs), on_send_result_(std::move(on_send_result)), on_drop_(std::move(on_drop)),
      interval_ms_(target_fps > 0 ? std::max<int64_t>(1, 1000 / target_fps) : 0),
      max_queue_(max_queue), states_(runs.size()) {
  for (size_t i = 0; i < states_.size(); ++i) {
    states_[i] = std::make_unique<State>();
    states_[i]->worker = std::thread([this, i]() { worker_loop(i); });
  }
}

ReleasePacer::~ReleasePacer() {
  stop();
}

int64_t ReleasePacer::now_ms_i64() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

bool ReleasePacer::enqueue(size_t idx, simaai::neat::Sample&& sample) {
  if (idx >= states_.size())
    return false;
  auto& st = *states_[idx];
  int64_t dropped_now = 0;

  {
    std::lock_guard<std::mutex> lk(st.mu);
    if (st.stop)
      return false;
    st.enqueued++;

    while (max_queue_ > 0 && st.queue.size() >= max_queue_) {
      st.queue.pop_front();
      st.dropped++;
      dropped_now++;
    }

    st.queue.push_back(std::move(sample));
    st.max_queue_depth =
        std::max<int64_t>(st.max_queue_depth, static_cast<int64_t>(st.queue.size()));
  }

  if (dropped_now > 0 && on_drop_) {
    on_drop_(idx, dropped_now);
  }

  st.cv.notify_one();
  return true;
}

void ReleasePacer::stop() {
  bool expected = false;
  if (!stopped_.compare_exchange_strong(expected, true))
    return;

  for (auto& st : states_) {
    if (!st)
      continue;
    {
      std::lock_guard<std::mutex> lk(st->mu);
      st->stop = true;
    }
    st->cv.notify_one();
  }

  for (auto& st : states_) {
    if (!st)
      continue;
    if (st->worker.joinable())
      st->worker.join();
  }
}

ReleasePacer::Stats ReleasePacer::stats(size_t idx) const {
  if (idx >= states_.size())
    return Stats{};
  const auto& st = *states_[idx];
  std::lock_guard<std::mutex> lk(st.mu);
  return Stats{st.enqueued, st.sent, st.dropped, st.send_fail, st.max_queue_depth};
}

void ReleasePacer::worker_loop(size_t idx) {
  auto& st = *states_[idx];
  while (true) {
    simaai::neat::Sample sample;
    {
      std::unique_lock<std::mutex> lk(st.mu);
      st.cv.wait(lk, [&]() { return st.stop || !st.queue.empty(); });
      if (st.stop && st.queue.empty())
        break;
      sample = std::move(st.queue.front());
      st.queue.pop_front();
    }

    if (interval_ms_ > 0) {
      int64_t now_ms = now_ms_i64();
      if (st.next_release_ms < 0)
        st.next_release_ms = now_ms;
      if (now_ms < st.next_release_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(st.next_release_ms - now_ms));
      }
      now_ms = now_ms_i64();
      st.next_release_ms = now_ms + interval_ms_;
    }

    bool ok = false;
    if (idx < runs_.size() && runs_[idx]) {
      ok = runs_[idx]->push(sample);
    }

    if (on_send_result_) {
      on_send_result_(idx, ok);
    }

    {
      std::lock_guard<std::mutex> lk(st.mu);
      if (ok) {
        st.sent++;
      } else {
        st.send_fail++;
      }
    }
  }
}

} // namespace simaai::neat::graph::strict_sync
