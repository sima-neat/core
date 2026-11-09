/**
 * @file
 * @ingroup pipeline
 * @brief `Result<T>` — a value, or why there is none: what `Run::pull(timeout)` returns.
 *
 * A `Result` carries one of the four outcomes of a pull (`PullStatus`): `Ok` with a value, or
 * `Timeout`, `Closed` or `Error` with a `PullError` that says why. Test it before reading the
 * value:
 *
 * @code
 *   auto r = run.pull("out", std::chrono::milliseconds(100));
 *   if (r) {
 *     use(*r);                                   // the Tensor
 *   } else if (r.status() == PullStatus::Closed) {
 *     // end of stream
 *   } else if (r.status() == PullStatus::Error) {
 *     log(r.error().code, r.error().message);    // a pipeline/ErrorCodes.h family
 *   }
 * @endcode
 *
 * @see Run::pull
 */
#pragma once

#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"

#include <optional>
#include <string>
#include <utility>

namespace simaai::neat {

/**
 * @brief A value of type `T`, or the outcome that left none (`Timeout`, `Closed`, `Error`).
 *
 * `status()` says which; `error()` carries the code and message of every outcome but `Ok`. Reading
 * the value of a `Result` that has none throws `NeatError` with the result's code
 * (`runtime.pull` when it has none). A default-constructed `Result` is an `Error` with no message.
 * @ingroup pipeline
 */
template <class T> class Result {
public:
  /// An `Error` result with no message.
  Result() = default;

  /// An `Ok` result holding `value`.
  static Result success(T value) {
    Result out;
    out.status_ = PullStatus::Ok;
    out.value_.emplace(std::move(value));
    return out;
  }

  /// A result without a value: `status` is `Timeout`, `Closed` or `Error` (`Ok` is taken as
  /// `Error`, since there is no value); `error` says why.
  static Result failure(PullStatus status, PullError error) {
    Result out;
    out.status_ = status == PullStatus::Ok ? PullStatus::Error : status;
    out.error_ = std::move(error);
    return out;
  }

  /// Which of the four outcomes this is.
  PullStatus status() const noexcept {
    return status_;
  }
  /// True iff the result holds a value (`status() == PullStatus::Ok`).
  bool ok() const noexcept {
    return status_ == PullStatus::Ok;
  }
  /// Same as `ok()`.
  explicit operator bool() const noexcept {
    return ok();
  }
  /// Why there is no value: code and message for `Timeout`, `Closed` and `Error`; empty for `Ok`.
  const PullError& error() const noexcept {
    return error_;
  }

  /// The value; throws `NeatError` when there is none.
  T& value() & {
    require_value();
    return *value_;
  }
  /// The value; throws `NeatError` when there is none.
  const T& value() const& {
    require_value();
    return *value_;
  }
  /// The value, moved out; throws `NeatError` when there is none.
  T&& value() && {
    require_value();
    return std::move(*value_);
  }
  /// The value; throws `NeatError` when there is none.
  T& operator*() & {
    return value();
  }
  /// The value; throws `NeatError` when there is none.
  const T& operator*() const& {
    return value();
  }
  /// The value, moved out; throws `NeatError` when there is none.
  T&& operator*() && {
    return std::move(*this).value();
  }
  /// Member access to the value; throws `NeatError` when there is none.
  T* operator->() {
    return &value();
  }
  /// Member access to the value; throws `NeatError` when there is none.
  const T* operator->() const {
    return &value();
  }

private:
  void require_value() const {
    if (!value_.has_value()) {
      throw_no_value();
    }
  }

  [[noreturn]] void throw_no_value() const {
    const char* outcome = status_ == PullStatus::Timeout  ? "Timeout"
                          : status_ == PullStatus::Closed ? "Closed"
                                                          : "Error";
    std::string what = std::string("Result::value: no value (") + outcome + ")";
    if (!error_.message.empty()) {
      what += ": " + error_.message;
    }
    GraphReport report = error_.report.value_or(GraphReport{});
    if (report.error_code.empty()) {
      report.error_code =
          error_.code.empty() ? std::string(error_codes::kRuntimePull) : error_.code;
    }
    if (report.repro_note.empty()) {
      report.repro_note = what;
    }
    std::string message = "[" + report.error_code + "] " + what;
    throw NeatError(std::move(message), std::move(report));
  }

  PullStatus status_ = PullStatus::Error;
  std::optional<T> value_;
  PullError error_;
};

} // namespace simaai::neat
