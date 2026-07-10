#include "test_main.h"

#include <sima_lmm/tool_call_parser.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using simaai::llima::ToolCallStreamParser;

std::vector<std::string> collect_content(const std::vector<std::string>& chunks) {
  ToolCallStreamParser parser;
  std::vector<std::string> content;
  for (const auto& chunk : chunks) {
    for (auto& event : parser.add(chunk, false)) {
      require(std::holds_alternative<ToolCallStreamParser::Content>(event),
              "expected only content event");
      content.push_back(std::move(std::get<ToolCallStreamParser::Content>(event).text));
    }
  }
  for (auto& event : parser.add("", true)) {
    require(std::holds_alternative<ToolCallStreamParser::Content>(event),
            "expected only content flush event");
    content.push_back(std::move(std::get<ToolCallStreamParser::Content>(event).text));
  }
  return content;
}

nlohmann::json collect_tool_calls(const std::vector<std::string>& chunks) {
  ToolCallStreamParser parser;
  nlohmann::json tool_calls = nullptr;
  for (const auto& chunk : chunks) {
    for (auto& event : parser.add(chunk, false)) {
      require(std::holds_alternative<ToolCallStreamParser::ToolCalls>(event),
              "expected only tool call event");
      tool_calls = std::move(std::get<ToolCallStreamParser::ToolCalls>(event).calls);
    }
  }
  for (auto& event : parser.add("", true)) {
    require(std::holds_alternative<ToolCallStreamParser::ToolCalls>(event),
            "expected only tool call flush event");
    tool_calls = std::move(std::get<ToolCallStreamParser::ToolCalls>(event).calls);
  }
  require(!tool_calls.is_null(), "missing tool call event");
  return tool_calls;
}

void require_content_chunks(const std::vector<std::string>& actual,
                            const std::vector<std::string>& expected) {
  require(actual == expected, "content chunk sequence mismatch");
}

void require_first_call(const nlohmann::json& tool_calls, const std::string& name,
                        const std::string& city) {
  require(tool_calls.is_array(), "tool calls must be an array");
  require(tool_calls.size() == 1U, "expected one tool call");
  const auto& call = tool_calls.at(0);
  require(call.at("type") == "function", "tool call type mismatch");
  require(call.at("function").at("name") == name, "tool call name mismatch");
  const auto args = nlohmann::json::parse(call.at("function").at("arguments").get<std::string>());
  require(args.at("city") == city, "tool call city mismatch");
}

void require_call_names(const nlohmann::json& tool_calls,
                        const std::vector<std::string>& expected_names) {
  require(tool_calls.is_array(), "tool calls must be an array");
  require(tool_calls.size() == expected_names.size(), "tool call count mismatch");
  for (std::size_t i = 0; i < expected_names.size(); ++i) {
    require(tool_calls.at(i).at("function").at("name") == expected_names.at(i),
            "tool call name mismatch");
  }
}

void require_argument(const nlohmann::json& tool_calls, const std::string& key,
                      const std::string& expected) {
  const auto args =
      nlohmann::json::parse(tool_calls.at(0).at("function").at("arguments").get<std::string>());
  require(args.at(key) == expected, "tool call argument mismatch");
}

} // namespace

RUN_TEST("unit_genai_tool_call_parser_test", ([] {
           require_content_chunks(collect_content({"There", " are", " 50", " states."}),
                                  {"There", " are", " 50", " states."});
           require_content_chunks(collect_content({"T", "h", "e", "re", " are", " facts."}),
                                  {"T", "h", "e", "re", " are", " facts."});
           require_content_chunks(collect_content({"<b", "old>", " text"}),
                                  {"<b", "old>", " text"});
           require_content_chunks(collect_content({"callout", ": this is just prose"}),
                                  {"callout", ": this is just prose"});

           require_first_call(
               collect_tool_calls({"ca", "ll:get_weather{city:", "<|\"|>To", "kyo<|\"|>}"}),
               "get_weather", "Tokyo");
           require_first_call(
               collect_tool_calls(
                   {"[{\"name\":\"get_", "weather\",\"arguments\":{\"city\":\"Tokyo\"}}]"}),
               "get_weather", "Tokyo");
           require_first_call(
               collect_tool_calls({"<tool_", "call>\n{\"name\":\"get_weather\","
                                               "\"arguments\":{\"city\":\"Tokyo\"}}",
                                   "\n</tool_call>"}),
               "get_weather", "Tokyo");
           require_first_call(
               collect_tool_calls({"{\"name\":\"get_weather\"",
                                   ",\"parameters\":{\"city\":\"Tokyo\"}}"}),
               "get_weather", "Tokyo");

           require_call_names(
               collect_tool_calls({"<tool_call>{\"name\":\"first\",\"arguments\":{}}",
                                   "</tool_call>",
                                   "<tool_call>{\"name\":\"second\",\"arguments\":{}}",
                                   "</tool_call>"}),
               {"first", "second"});
           require_call_names(
               collect_tool_calls({"call:first{}", " call:second{}"}), {"first", "second"});

           require_first_call(
               collect_tool_calls(
                   {"[TOOL_CALLS]\n[{\"name\":\"get_weather\",",
                    "\"arguments\":{\"city\":\"Tokyo\"}}]"}),
               "get_weather", "Tokyo");

           const auto quoted_brace = collect_tool_calls(
               {"{\"name\":\"search\",\"parameters\":{\"query\":\"a}b\\\\c\"}}"});
           require_call_names(quoted_brace, {"search"});
           require_argument(quoted_brace, "query", "a}b\\c");

           require_content_chunks(collect_content({"{\"answer\":", "\"There are 50 states.\"}"}),
                                  {"{\"answer\":\"There are 50 states.\"}"});
           require_content_chunks(collect_content({"{\"name\":\"Alice\",\"age\":30}"}),
                                  {"{\"name\":\"Alice\",\"age\":30}"});
           require_content_chunks(
               collect_content({"{\"name\":\"weather\",\"arguments\":\"not-json\"}"}),
               {"{\"name\":\"weather\",\"arguments\":\"not-json\"}"});
           require_content_chunks(collect_content({"[]"}), {"[]"});
           require_content_chunks(
               collect_content(
                   {"<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call> more"}),
               {"<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call> more"});

           const std::string weather_call =
               "{\"name\":\"weather\",\"arguments\":{\"city\":\"Tokyo\"}}";
           require(simaai::llima::try_parse_tool_calls(weather_call, {"search"}).is_null(),
                   "undeclared tool name should be rejected");
           require(!simaai::llima::try_parse_tool_calls(weather_call, {"weather"}).is_null(),
                   "declared tool name should be accepted");

           ToolCallStreamParser restricted_parser({"search"});
           const auto rejected_events = restricted_parser.add(weather_call, true);
           require(rejected_events.size() == 1U, "rejected call should fall back to content");
           require(std::holds_alternative<ToolCallStreamParser::Content>(rejected_events.at(0)),
                   "undeclared streamed tool should be content");
         }));
