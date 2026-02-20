#include "pipeline/Session.h"

#include <iostream>
#include <type_traits>
#include <utility>

template <typename T, typename = void>
struct has_session_build_input_runoptions_only : std::false_type {};

template <typename T>
struct has_session_build_input_runoptions_only<
    T, std::void_t<decltype(std::declval<T&>().build(
           std::declval<const simaai::neat::Tensor&>(),
           std::declval<const simaai::neat::RunOptions&>()))>> : std::true_type {};

template <typename T, typename = void> struct has_set_frame_callback : std::false_type {};

template <typename T>
struct has_set_frame_callback<T, std::void_t<decltype(&T::set_frame_callback)>> : std::true_type {};

int main() {
  using simaai::neat::Run;
  using simaai::neat::Session;
  using simaai::neat::Tensor;

  static_assert(
      std::is_same_v<decltype(std::declval<Session&>().build(std::declval<const Tensor&>())), Run>,
      "Session::build(Tensor) should resolve to Run");
  static_assert(!has_session_build_input_runoptions_only<Session>::value,
                "Session::build(Tensor, RunOptions) should not exist");
  static_assert(!has_set_frame_callback<Session>::value,
                "Session::set_frame_callback should not exist");

  std::cout << "[OK] unit_session_api_surface_test passed\n";
  return 0;
}
