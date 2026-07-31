#include "nodes/common/Caps.h"
#include "pipeline/graph/GraphDetail.h"
#include "pipeline/internal/GstLaunchBindings.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>

#include <unistd.h>

RUN_TEST(
    "unit_gst_launch_bindings_test", ([] {
      using namespace simaai::neat::pipeline_internal::gst_launch;

      {
        const std::string launch = "identity name=plain ! identity name = \"quoted\" ! "
                                   "identity name='literal' ! identity name=escaped\\ value "
                                   "encoding-name=H264";
        const Analysis analysis = analyze(launch);
        require(analysis.complete, "ordinary assignments should analyze completely");
        const auto names = explicit_name_bindings(analysis);
        require(names.size() == 4U, "expected four exact name assignments");
        require(names[0]->canonical_value == "plain", "unquoted name canonicalization failed");
        require(names[1]->canonical_value == "quoted",
                "double-quoted name canonicalization failed");
        require(names[2]->canonical_value == "'literal'",
                "single quotes must remain part of the GStreamer value");
        require(names[3]->canonical_value == "escaped value",
                "escaped whitespace canonicalization failed");

        const RewriteResult single_quote_rewrite =
            rewrite(launch, analysis, {{"'literal'", "literal_renamed"}});
        const Analysis rewritten_analysis = analyze(single_quote_rewrite.text);
        const auto rewritten_names = explicit_name_bindings(rewritten_analysis);
        require(rewritten_names.size() == 4U &&
                    rewritten_names[2]->canonical_value == "literal_renamed",
                "rewritten single-quoted values must preserve canonical target semantics");
      }

      {
        const std::string launch =
            "fakesrc location=\"https://host/path?name=url\" ! "
            "video/x-raw(memory:DMABuf),format=(string){NV12,I420},name=caps_only;"
            "video/x-raw,framerate=(fraction)[1/1,60/1],name=also_caps ! "
            "capsfilter caps=video/x-raw,name=property_value ! "
            "identity name=real";
        const Analysis analysis = analyze(launch);
        const auto names = explicit_name_bindings(analysis);
        require(names.size() == 1U && names.front()->canonical_value == "real",
                "URL, caps, and assignment values must not create false name bindings");
      }

      {
        const std::string launch = "( appsrc name=mysrc is-live=true ! "
                                   "video/x-raw,format=NV12,width=640,height=480,name=caps_only ); "
                                   "identity name=after_caps ! video/x-raw,name=terminal_caps";
        const Analysis analysis = analyze(launch);
        const auto names = explicit_name_bindings(analysis);
        require(analysis.complete,
                "terminal caps at a bin or chain boundary are valid Gst launch syntax");
        require(names.size() == 2U && names[0]->canonical_value == "mysrc" &&
                    names[1]->canonical_value == "after_caps",
                "terminal caps must not hide later chains or expose caps fields as element names");
      }

      {
        const std::string launch = "tee name=t t. ! queue name=q";
        const Analysis analysis = analyze(launch);
        require(analysis.references.size() == 1U &&
                    analysis.references.front().canonical_element_name == "t",
                "bare named-element references should be identified");
        const RewriteResult rewritten = rewrite(launch, analysis, {{"t", "renamed_tee"}});
        require_contains(rewritten.text, "name=renamed_tee",
                         "bare-reference declaration was not rewritten");
        require_contains(rewritten.text, "renamed_tee. !",
                         "bare named-element reference was not rewritten");
      }

      {
        const std::string launch = "simaai_sampledemux name=demux "
                                   "demux.bbox ! queue ! render.sink_0 "
                                   "demux.image ! queue ! render.sink_1 "
                                   "neatrender name=render stage-id=render op-buff-name=demux "
                                   "next-element=CVU";
        const Analysis analysis = analyze(launch);
        require(analysis.references.size() == 4U, "all named-pad references should be identified");

        const NameMapping mapping = {
            {"demux", "x_demux_y"}, {"render", "x_render_y"}, {"CVU", "must_not_replace_selector"}};
        static constexpr std::string_view aliases[] = {"stage-id", "op-buff-name"};
        const RewriteResult rewritten = rewrite(launch, analysis, mapping, aliases);
        require(rewritten.complete, "valid UdpOutputGroupG fragment rewrite was incomplete");
        require_contains(rewritten.text, "name=x_demux_y", "demux declaration was not rewritten");
        require_contains(rewritten.text, "name=x_render_y", "render declaration was not rewritten");
        require_contains(rewritten.text, "x_demux_y.bbox",
                         "demux.bbox reference was not rewritten");
        require_contains(rewritten.text, "x_demux_y.image",
                         "demux.image reference was not rewritten");
        require_contains(rewritten.text, "x_render_y.sink_0",
                         "render.sink_0 reference was not rewritten");
        require_contains(rewritten.text, "x_render_y.sink_1",
                         "render.sink_1 reference was not rewritten");
        require_contains(rewritten.text, "stage-id=x_render_y",
                         "documented stage-id alias was not rewritten");
        require_contains(rewritten.text, "op-buff-name=x_demux_y",
                         "documented op-buff-name alias was not rewritten");
        require_contains(rewritten.text, "next-element=CVU",
                         "next-element selector must not be rewritten generically");
      }

      {
        simaai::neat::NameTransform transform;
        transform.prefix = "pre_";
        transform.suffix = "_suf";
        const auto node =
            simaai::neat::nodes::Custom("simaai_sampledemux name=demux demux.bbox ! render.sink_0 "
                                        "neatrender name=render");
        const simaai::neat::NodeFragment fragment =
            simaai::neat::make_node_fragment(node, 0, transform);
        require(fragment.element_names.size() == 2U,
                "Custom multi-element fragment should report every explicit name");
        require_contains(fragment.fragment, "pre_demux_suf.bbox",
                         "Custom rewrite should keep declaration/reference mapping coherent");
        require_contains(fragment.fragment, "pre_render_suf.sink_0",
                         "Custom render reference should follow transformed declaration");
      }

      {
        const auto node = simaai::neat::nodes::Custom(
            "fakesrc location=https://example.invalid/(!;)?name=not_a_binding");
        require_contains(node->backend_fragment(7), "name=n7_fakesrc",
                         "a URL property must not suppress deterministic naming for a simple "
                         "single-element Custom node");
        const auto names = node->element_names(7);
        require(names.size() == 1U && names.front() == "n7_fakesrc",
                "simple URL Custom node should report its generated deterministic name");
      }

      {
        const std::filesystem::path config_path =
            std::filesystem::temp_directory_path() /
            ("gst_launch_binding_config_" + std::to_string(::getpid()) + ".json");
        {
          std::ofstream config(config_path);
          config << R"({"input_buffers":[{"name":"old"}]})";
        }
        const std::string original_path = config_path.string();
        auto node = simaai::neat::nodes::Custom("fakefactory config=\"" + original_path +
                                                "\" note=\"" + original_path + "\"");
        require(node->wire_input_names({"new_input"}, "test"),
                "Custom config wiring should rewrite a matching config assignment");
        const std::string rewritten = node->backend_fragment(0);
        require(rewritten.find("config=\"" + original_path + "\"") == std::string::npos,
                "config assignment should point to the rewritten temporary JSON");
        require_contains(rewritten, "note=\"" + original_path + "\"",
                         "equal text outside the config assignment must remain unchanged");
        std::filesystem::remove(config_path);
      }

      {
        const Analysis malformed = analyze("identity name=\"unterminated");
        require(!malformed.complete, "unterminated quoted assignment must be incomplete");
        const RewriteResult unchanged =
            rewrite("identity name=\"unterminated", malformed, {{"x", "y"}});
        require(!unchanged.complete && unchanged.text == "identity name=\"unterminated",
                "incomplete syntax must never be partially rewritten");
      }

      {
        std::mt19937 generator(0x458U);
        static constexpr char kAlphabet[] =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 =!:.\\\"'()[]{};,/-_";
        for (std::size_t corpus_index = 0; corpus_index < 5'000U; ++corpus_index) {
          const std::size_t length = generator() % 192U;
          std::string input;
          input.reserve(length);
          for (std::size_t i = 0; i < length; ++i) {
            input.push_back(kAlphabet[generator() % (sizeof(kAlphabet) - 1U)]);
          }
          const Analysis analysis = analyze(input);
          for (const auto& assignment : analysis.assignments) {
            require(assignment.token_span.begin <= assignment.value_span.begin &&
                        assignment.value_span.end <= assignment.token_span.end &&
                        assignment.token_span.end <= input.size(),
                    "fuzzed assignment returned an out-of-bounds span");
          }
          for (const auto& reference : analysis.references) {
            require(reference.token_span.begin <= reference.element_span.begin &&
                        reference.element_span.end <= reference.token_span.end &&
                        reference.token_span.end <= input.size(),
                    "fuzzed reference returned an out-of-bounds span");
          }

          const RewriteResult no_op = rewrite(input, analysis, {});
          require(no_op.text == input, "empty-map rewrite must be byte-for-byte idempotent");
          if (!analysis.complete) {
            const RewriteResult fail_closed = rewrite(input, analysis, {{"x", "y"}});
            require(!fail_closed.complete && fail_closed.text == input && !fail_closed.changed,
                    "malformed fuzz input must fail closed without a partial edit");
          }
        }
      }

      {
        std::string large;
        large.reserve(1'000'000U);
        for (std::size_t i = 0; i < 40'000U; ++i) {
          large += "identity name=n" + std::to_string(i) + " ! ";
        }
        const auto start = std::chrono::steady_clock::now();
        const Analysis analysis = analyze(large);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        require(analysis.complete && explicit_name_bindings(analysis).size() == 40'000U,
                "large linear corpus should retain every explicit assignment");
        require(elapsed < std::chrono::seconds(5),
                "launch analysis exceeded the linear-growth complexity budget");
      }
    }));
