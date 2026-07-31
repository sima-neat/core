#include "gst/GstInit.h"
#include "pipeline/internal/GstParseLaunch.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/GstLaunchBindings.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <array>
#include <string>

namespace {

class ScopedGObjectCriticalSilencer {
public:
  ScopedGObjectCriticalSilencer()
      : handler_(g_log_set_handler(
            "GLib-GObject", G_LOG_LEVEL_CRITICAL,
            +[](const gchar*, GLogLevelFlags, const gchar*, gpointer) {}, nullptr)) {}

  ~ScopedGObjectCriticalSilencer() {
    g_log_remove_handler("GLib-GObject", handler_);
  }

  ScopedGObjectCriticalSilencer(const ScopedGObjectCriticalSilencer&) = delete;
  ScopedGObjectCriticalSilencer& operator=(const ScopedGObjectCriticalSilencer&) = delete;

private:
  guint handler_ = 0;
};

} // namespace

RUN_TEST(
    "unit_gst_parse_launch_audit_test", ([] {
      using namespace simaai::neat;
      gst_init_once();

      auto parse = [](std::string launch) {
        BuildResult build;
        build.pipeline_string = std::move(launch);
        build.diag = std::make_shared<DiagCtx>();
        return session_build_parse_pipeline_or_throw(build, "parse-audit-test");
      };

      {
        GstElement* pipeline = parse("identity name=first identity name=second");
        require(pipeline != nullptr && GST_IS_BIN(pipeline),
                "unique explicit names should parse as one pipeline segment");
        gst_object_unref(pipeline);
      }

      {
        GstElement* pipeline = parse("identity name='single_literal' identity name=companion");
        const auto inventory = gst::inventory_elements(pipeline);
        bool found_literal_quotes = false;
        for (const auto& object : inventory) {
          found_literal_quotes = found_literal_quotes || object.short_name == "'single_literal'";
        }
        require(found_literal_quotes,
                "target GStreamer should retain single quotes as literal name characters");
        gst_object_unref(pipeline);
      }

      {
        struct DifferentialCase {
          const char* assignment;
          const char* expected;
        };
        static constexpr std::array<DifferentialCase, 3> kCases = {{
            {"name=plain_differential", "plain_differential"},
            {"name=\"double\\ quoted\"", "double quoted"},
            {"name=escaped\\ value", "escaped value"},
        }};
        for (std::size_t i = 0; i < kCases.size(); ++i) {
          const std::string launch = std::string("identity ") + kCases[i].assignment +
                                     " identity name=differential_companion_" + std::to_string(i);
          const auto analysis = pipeline_internal::gst_launch::analyze(launch);
          const auto bindings = pipeline_internal::gst_launch::explicit_name_bindings(analysis);
          require(analysis.complete && !bindings.empty() &&
                      bindings.front()->canonical_value == kCases[i].expected,
                  "binding analyzer canonicalization disagreed with the differential fixture");
          GstElement* pipeline = parse(launch);
          const auto inventory = gst::inventory_elements(pipeline);
          bool native_match = false;
          for (const auto& object : inventory) {
            native_match = native_match || object.short_name == kCases[i].expected;
          }
          require(native_match,
                  "binding analyzer canonicalization disagreed with target GStreamer");
          gst_object_unref(pipeline);
        }
      }

      {
        bool threw = false;
        try {
          GstElement* pipeline = parse("identity name=duplicate identity name=duplicate");
          if (pipeline) {
            gst_object_unref(pipeline);
          }
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kPipelineShape,
                  "duplicate explicit names should be a pipeline-shape error");
          require_contains(error.what(), "duplicate explicit GStreamer element name",
                           "duplicate diagnostic should explain the binding collision");
        }
        require(threw, "duplicate explicit names must fail without Graph::validate()");
      }

      {
        bool threw = false;
        ScopedGObjectCriticalSilencer silence_expected_duplicate_property_warning;
        try {
          GstElement* pipeline = parse("identity name=first name=second identity");
          if (pipeline) {
            gst_object_unref(pipeline);
          }
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kPipelineShape,
                  "a declaration dropped by repeated name properties should be a shape error");
          require_contains(error.what(), "did not survive construction exactly once",
                           "declaration-survival diagnostic should explain the dropped name");
        }
        require(threw, "every explicit declaration must survive native construction");
      }

      {
        bool threw = false;
        try {
          GstElement* pipeline = parse("definitely_not_a_gstreamer_factory name=missing");
          if (pipeline) {
            gst_object_unref(pipeline);
          }
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kParseLaunch,
                  "native plugin/syntax failures should remain parse-launch errors");
        }
        require(threw, "a missing factory must fail through native parsing");
      }

      {
        Graph graph;
        graph.custom("fakesrc name=build_duplicate ! identity name=build_duplicate ! fakesink",
                     InputRole::Source);
        bool threw = false;
        try {
          Run run = graph.build();
          run.close();
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kPipelineShape,
                  "automatic Graph::build validation should preserve the shape error code");
          require_contains(error.what(), "Node fragment node[0]",
                           "build diagnostics should attribute collisions to rendered Nodes");
        }
        require(threw, "Graph::build must reject duplicate launch names without validate()");
      }

      {
        bool threw = false;
        try {
          session_build_validate_explicit_launch_names_or_throw(
              "( identity name=rtsp_duplicate identity name=rtsp_duplicate )",
              "rtsp-precheck-test");
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kPipelineShape,
                  "RTSP explicit collision should be a pipeline-shape error");
        }
        require(threw, "RTSP delayed-parse path must reject explicit collisions synchronously");
      }

      {
        bool threw = false;
        try {
          GstElement* pipeline = parse("bin.( name=left identity name=nested ) "
                                       "bin.( name=right identity name=nested )");
          if (pipeline) {
            gst_object_unref(pipeline);
          }
        } catch (const NeatError& error) {
          threw = true;
          require(error.report().error_code == error_codes::kPipelineShape,
                  "nested short-name ambiguity should be a pipeline-shape error");
        }
        require(threw, "equal short names in nested bins must fail under segment-global policy");
      }

      {
        // Scope is one native parse. Reusing a name in a separately parsed segment is safe.
        GstElement* first = parse("identity name=reused identity");
        GstElement* second = parse("identity name=reused identity");
        require(first != nullptr && second != nullptr,
                "independently parsed segments may reuse the same short name");
        gst_object_unref(first);
        gst_object_unref(second);
      }

      {
        GstElement* pipeline = gst_pipeline_new("inventory_root");
        GstElement* left = gst_bin_new("left_bin");
        GstElement* right = gst_bin_new("right_bin");
        GstElement* left_child = gst_element_factory_make("identity", "nested_duplicate");
        GstElement* right_child = gst_element_factory_make("identity", "nested_duplicate");
        require(pipeline && left && right && left_child && right_child,
                "failed to construct recursive inventory fixture");
        require(gst_bin_add(GST_BIN(left), left_child) &&
                    gst_bin_add(GST_BIN(right), right_child) &&
                    gst_bin_add(GST_BIN(pipeline), left) && gst_bin_add(GST_BIN(pipeline), right),
                "failed to assemble recursive inventory fixture");
        const auto inventory = gst::inventory_elements(pipeline);
        std::size_t duplicates = 0;
        for (const auto& object : inventory) {
          if (object.short_name == "nested_duplicate") {
            ++duplicates;
            require(!object.object_path.empty() && !object.parent_path.empty(),
                    "recursive inventory should retain object and parent paths");
          }
        }
        require(duplicates == 2U,
                "recursive inventory must expose equal short names in separate nested bins");
        gst_object_unref(pipeline);
      }
    }));
