#include "pipeline/graph/GraphDetail.h"
#include "test_main.h"
#include "test_utils.h"

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string random_name(std::mt19937& rng, int min_len = 3, int max_len = 12) {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz"
                                  "0123456789";
  std::uniform_int_distribution<int> len_dist(min_len, max_len);
  std::uniform_int_distribution<int> ch_dist(0, static_cast<int>(sizeof(kAlphabet) - 2));

  const int len = len_dist(rng);
  std::string out;
  out.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    out.push_back(kAlphabet[ch_dist(rng)]);
  }
  return out;
}

} // namespace

RUN_TEST("property_pipeline_string_test", ([] {
           using simaai::neat::rewrite_fragment_names;

           std::mt19937 rng(1337u);
           std::uniform_int_distribution<int> count_dist(1, 4);

           for (int iter = 0; iter < 250; ++iter) {
             const int names_count = count_dist(rng);
             std::vector<std::string> base_names;
             base_names.reserve(static_cast<std::size_t>(names_count));
             for (int i = 0; i < names_count; ++i) {
               base_names.push_back("n" + std::to_string(i) + "_" + random_name(rng));
             }

             std::string fragment;
             for (int i = 0; i < names_count; ++i) {
               if (i)
                 fragment += " ! ";
               fragment += "identity name=" + base_names[i] + " op-buff-name='" + base_names[i] +
                           "' next-element=\"" + base_names[i] + "\"";
             }

             std::unordered_map<std::string, std::string> mapping;
             for (const auto& n : base_names) {
               mapping.emplace(n, "x_" + n + "_y");
             }

             const std::string once = rewrite_fragment_names(fragment, mapping);
             const std::string twice = rewrite_fragment_names(once, mapping);

             require(once == twice,
                     "rewrite_fragment_names should be idempotent under repeated application");

             for (const auto& kv : mapping) {
               require_contains(once, kv.second,
                                "rewritten fragment missing remapped element name");
             }
           }

           // Invalid quoting should not crash the rewriter.
           {
             const std::string malformed = "identity name='unterminated ! queue name=q0";
             std::unordered_map<std::string, std::string> mapping{{"q0", "q1"}};
             const std::string out = rewrite_fragment_names(malformed, mapping);
             require(!out.empty(), "rewrite_fragment_names should return a non-empty string");
           }
         }));
