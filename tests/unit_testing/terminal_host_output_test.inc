// Actual public frontend allocation, not a forged allocator tag/certificate.
void terminal_host_output_lifetime(bool dma_denied = false) {
  namespace neat = simaai::neat;
  namespace internal = neat::pipeline_internal;
  using namespace sima_test::output_contract;
  // Pin the existing test policy, restoring the caller environment afterward.
  struct QueueCap {
    std::optional<std::string> prior;
    QueueCap() {
      if (const char* value = std::getenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP"))
        prior = value;
      setenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", "1", 1);
    }
    ~QueueCap() {
      if (prior)
        setenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", prior->c_str(), 1);
      else
        unsetenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP");
    }
  } queue_cap;

  std::shared_ptr<GstElement> pipeline(gst_pipeline_new("terminal-host-core"), gst_object_unref);
  auto* source = gst_element_factory_make("appsrc", "source");
  auto* stage = gst_element_factory_make("neatkernelrunner", "cast");
  auto* sink = gst_element_factory_make("appsink", "sink");
  require(source && stage && sink, "terminal host pipeline factories");
  gst_bin_add_many(GST_BIN(pipeline.get()), source, stage, sink, nullptr);
  auto* sink_pad = gst_element_request_pad_simple(stage, "sink_0");
  auto* source_pad = gst_element_get_static_pad(source, "src");
  require(sink_pad && source_pad && gst_pad_link(source_pad, sink_pad) == GST_PAD_LINK_OK,
          "terminal host source link");
  gst_object_unref(source_pad);
  gst_object_unref(sink_pad);
  require(gst_element_link(stage, sink), "terminal host sink link");
  g_object_set(sink, "sync", FALSE, "async", FALSE, "max-buffers", 1U, "drop", FALSE, nullptr);
  auto request = cast_request(1, true);
  request.options.output_pool_capacity = 6;
  gst_app_src_set_caps(GST_APP_SRC(source), request.publication.input_caps.get());
  std::array<gs::StageAttachment, 1> attachments{{{stage, std::move(request)}}};
  auto failure = gs::prepare_and_attach_tensor_stages(
      pipeline.get(), attachments, {}, gs::CompactFrameReaderContract::StageFrameV2, {stage, sink});
  require_fixture(!failure, "terminal host frontend admission", failure ? failure->message : "");
  struct Stop {
    GstElement* pipeline;
    ~Stop() {
      gst_element_set_state(pipeline, GST_STATE_NULL);
    }
  } stop{pipeline.get()};
  require(gst_element_set_state(pipeline.get(), GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "terminal host pipeline start");
  auto* input = gst_buffer_new_allocate(nullptr, 8, nullptr);
  const std::array<float, 2> values{1.25f, -2.5f};
  require(input && gst_buffer_fill(input, 0, values.data(), 8) == 8, "terminal host input bytes");
  // Direct physical input carries an independent, complete public tensor contract.
  simaai::gst::TensorBufferPublishContract contract;
  contract.stage_key = "host-input";
  contract.physical_outputs.push_back({0, 8, "input0"});
  simaai::gst::TensorBufferPublishLogicalOutput logical;
  logical.logical_index = logical.physical_index = logical.memory_index = 0;
  logical.backend_output_index = logical.route_slot = 0;
  logical.logical_name = logical.backend_name = logical.segment_name = "input0";
  logical.size_bytes = 8;
  logical.dtype = SIMA_TENSOR_SET_DTYPE_FP32_V1;
  logical.shape = {2};
  logical.stride_bytes = {4};
  contract.logical_outputs.push_back(logical);
  simaai::gst::TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require_fixture(
      simaai::gst::tensor_buffer_prepare_meta_template_from_contract(contract, &prepared, &error),
      "terminal host input contract", error);
  require_fixture(simaai::gst::tensor_buffer_attach_prepared_meta(input, prepared, &error),
                  "terminal host input publication", error);
  std::shared_ptr<GstBuffer> input_template(gst_buffer_ref(input), gst_buffer_unref);
  require(gst_app_src_push_buffer(GST_APP_SRC(source), input) == GST_FLOW_OK,
          "terminal host input push");
  std::shared_ptr<GstSample> raw(gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 5 * GST_SECOND),
                                 gst_sample_unref);
  require(bool(raw), "terminal host output timeout");
  neat::InputStream::State state;
  state.opt.preserve_framework_output = true;
  internal::reset_tensor_io_stats();
  auto output =
      neat::output_from_sample_stream(raw.get(), "terminal host Auto", true, nullptr, &state);
  require(output.tensors.size() == 1 && !output.owned &&
              internal::snapshot_tensor_io_stats().tensor_copy_count == 0,
          "initial Auto materialization copied certified host");
  auto& tensor = output.tensors.front();
  require(!internal::tensor_has_dmabuf_memory(tensor) &&
              internal::tensor_has_certified_terminal_host_memory(tensor),
          "terminal host was relabeled DMA or lost its generation");
  const auto oracle = [](const neat::Tensor& t) {
    const auto mapping = t.storage->map(neat::MapMode::Read);
    const std::array<std::uint16_t, 2> expected{0x3fa0, 0xc020};
    require(mapping.data && t.byte_offset >= 0 &&
                static_cast<std::size_t>(t.byte_offset) + 4 <= mapping.size_bytes &&
                std::memcmp(static_cast<const std::byte*>(mapping.data) + t.byte_offset,
                            expected.data(), 4) == 0,
            "terminal host Cast independent BF16 oracle");
  };
  oracle(tensor);
  // A numerical rank-zero view is one element, not an unbound/opaque carrier.
  // Use the actual framework allocation above: no forged certificate or allocator tag.
  auto scalar = tensor;
  scalar.shape.clear();
  scalar.strides_bytes.clear();
  scalar.byte_offset = 2;
  require(internal::tensor_has_certified_terminal_host_memory(scalar),
          "certified scalar lost its one-element host span");
  auto scalar_output = neat::sample_from_tensors(neat::TensorList{scalar});
  internal::reset_tensor_io_stats();
  internal::materialize_owned_output(scalar_output, true);
  require(scalar_output.tensors.front().storage == scalar.storage &&
              internal::snapshot_tensor_io_stats().tensor_copy_count == 0,
          "Auto copied a certified scalar instead of retaining it");
  auto scalar_outside = scalar;
  scalar_outside.byte_offset = 4;
  require(!internal::tensor_has_certified_terminal_host_memory(scalar_outside),
          "scalar span outside the selected framework allocation was accepted");
  auto opaque_scalar = scalar;
  opaque_scalar.semantic.byte_stream = neat::ByteStreamSpec{};
  require(!internal::tensor_has_certified_terminal_host_memory(opaque_scalar) &&
              !internal::tensor_has_certified_terminal_host_memory(neat::Tensor{}),
          "opaque or unbound empty shape was certified as a scalar");
  // The query is selected-span authority, never sibling, wrapper, or allocator-name authority.
  auto outside = tensor;
  outside.byte_offset = std::numeric_limits<std::int64_t>::max();
  require(!internal::tensor_has_certified_terminal_host_memory(outside),
          "host span overflow accepted");
  auto ambiguous = tensor;
  ambiguous.route.memory_index = 9;
  require(!internal::tensor_has_certified_terminal_host_memory(ambiguous),
          "foreign memory index accepted");
  auto foreign = frame(sima_test::output_contract::contract(false), {false}, 51);
  neat::InputStream::State foreign_state;
  auto foreign_output = neat::output_from_sample_stream(foreign.sample.get(), "foreign CPU", false,
                                                        nullptr, &foreign_state);
  require(!internal::sample_has_certified_terminal_host_memory(foreign_output),
          "foreign CPU certified");
  auto mixed = neat::sample_from_tensors(neat::TensorList{tensor, foreign_output.tensors.front()});
  for (bool requested_copy : {false, true}) {
    const auto decision =
        internal::output_queue_memory_decision(mixed, requested_copy, true, false);
    require(!decision.has_dma && !decision.only_framework && decision.preserve_framework &&
                decision.permits_pressure_copy() && decision.copy_when_latched(true),
            "mixed host must protect Block while copying its foreign sibling");
    const auto only_host =
        internal::output_queue_memory_decision(output, requested_copy, true, false);
    require(only_host.only_framework && only_host.preserve_framework &&
                !only_host.permits_pressure_copy(),
            "host-only reliability policy");
  }
  const auto strict = internal::output_queue_memory_decision(mixed, false, true, true);
  require(!strict.copy_when_latched(true), "explicit ZeroCopy policy was overridden");
  const auto requested_owned = internal::output_queue_memory_decision(mixed, true, false, false);
  require(!requested_owned.preserve_framework && requested_owned.copy_when_latched(true),
          "Owned must not preserve the certified host sibling");
  auto gate = std::make_shared<internal::HolderLoanGate>(4);
  require_fixture(internal::attach_zero_copy_loan_to_sample(mixed, gate, &error), "host loan",
                  error);
  const auto credit_namespace = internal::next_realtime_frame_credit_namespace();
  auto lane = internal::make_realtime_frame_credit_lane(1);
  require(lane->gate->try_acquire(), "real-time fixture credit acquire");
  require(internal::register_realtime_frame_credit(credit_namespace, "host-mixed", 7, lane),
          "real-time fixture credit register");
  struct CreditCleanup {
    std::uint64_t id;
    ~CreditCleanup() {
      internal::release_all_registered_realtime_frame_credits(id, "test-cleanup");
    }
  } credit_cleanup{credit_namespace};
  internal::attach_realtime_frame_credit_to_sample(
      mixed, {credit_namespace, "host-mixed", 7, -1, -1, false});
  const auto acquired = gate->inflight();
  require(acquired > 0, "host loan was not acquired");
  if (dma_denied) {
    // Separate device-denied case group. It does not qualify the positive
    // Owned/mixed-copy assertions below, which remain mandatory with real DMA.
    for (std::size_t queued : {0U, 1U}) {
      auto item = output;
      internal::reset_tensor_io_stats();
      neat::graph::maybe_force_copy_for_backpressure(item, queued, "host-only-test", 0, true);
      require(internal::sample_has_certified_terminal_host_memory(item) && !item.owned &&
                  internal::snapshot_tensor_io_stats().tensor_copy_count == 0,
              "host-only pressure must not allocate DMA");
    }
    auto owned_negative = output;
    bool rejected = false;
    try {
      internal::materialize_owned_output(owned_negative, false);
    } catch (const std::runtime_error& error) {
      rejected = std::string_view(error.what()).starts_with("Owned output allocation failed:");
    }
    require(rejected && !owned_negative.owned &&
                internal::sample_has_certified_terminal_host_memory(owned_negative),
            "denied Owned allocation must fail without a host fallback");
    internal::release_realtime_frame_credits_for_sample(mixed, "test-denied-explicit-pull");
    require(lane->gate->inflight() == 0 && lane->released_by_output.load() == 1,
            "device-denied fixture credit release");
  } else {
    // Both queue sizes exercise the same graph queue helper without changing its threshold.
    for (std::size_t queued : {0U, 1U}) {
      auto item = mixed;
      internal::reset_tensor_io_stats();
      neat::graph::maybe_force_copy_for_backpressure(item, queued, "terminal-host-test", 0, true);
      require(internal::tensor_has_certified_terminal_host_memory(item.tensors.front()) &&
                  !item.owned,
              "queue pressure copied host sibling or claimed full ownership");
      require(internal::snapshot_tensor_io_stats().tensor_copy_count == queued,
              "queue0/queue1 must respectively retain/copy only the foreign payload");
      require(gate->inflight() == acquired,
              "partial materialization returned retained holder credit");
    }
    internal::reset_tensor_io_stats();
    internal::materialize_owned_output(mixed, true);
    require(!mixed.owned && internal::snapshot_tensor_io_stats().tensor_copy_count == 1 &&
                internal::tensor_has_certified_terminal_host_memory(mixed.tensors.front()) &&
                internal::tensor_has_dmabuf_memory(mixed.tensors.back()),
            "mixed Auto must retain host and copy only foreign CPU");
    require(gate->inflight() == acquired && lane->gate->inflight() == 1 &&
                lane->released_by_output.load() == 0,
            "partial copy returned producer loan or registered credit early");
    internal::release_realtime_frame_credits_for_sample(mixed, "test-explicit-pull");
    require(lane->gate->inflight() == 0 && lane->released_by_output.load() == 1 &&
                lane->gate->overrelease() == 0,
            "registered mixed output credit did not release once");
    auto owned = output;
    internal::materialize_owned_output(owned, false);
    require(owned.owned && internal::tensor_has_dmabuf_memory(owned.tensors.front()) &&
                !internal::tensor_has_certified_terminal_host_memory(owned.tensors.front()),
            "Owned did not allocate independent DMA storage");
    oracle(owned.tensors.front());
  }
  // Exercise the actual asynchronous Run callback with real host output. A
  // queue of one requests Block. Exact decision branches are tested above; the
  // pull counter below witnesses decoding, not a queue-decision rendezvous.
  neat::InputStreamOptions stream_options;
  stream_options.copy_output = false;
  stream_options.preserve_framework_output = true;
  stream_options.public_output_contract = true;
  stream_options.holder_loan_credits = 4;
  stream_options.enable_timings = true;
  neat::SampleSpec sample_spec;
  sample_spec.kind = neat::SampleMediaKind::Tensor;
  sample_spec.media_type = "application/x-sima-tensor";
  sample_spec.dtype = neat::TensorDType::BFloat16;
  sample_spec.shape = {2};
  auto stream = neat::InputStream::create(
      GST_ELEMENT(gst_object_ref(pipeline.get())), nullptr, GST_ELEMENT(gst_object_ref(sink)),
      sample_spec, {}, stream_options, std::make_shared<internal::DiagCtx>(), {});
  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Balanced;
  run_options.output_memory = neat::OutputMemory::Auto;
  run_options.queue_depth = 1;
  run_options.overflow_policy = neat::OverflowPolicy::Block;
  auto core = neat::runtime::RunCore::start_single_pipeline(std::move(stream), run_options,
                                                            stream_options, neat::RunMode::Async);
  const auto wait_until = [&](auto predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(predicate(), message);
  };
  for (unsigned sequence = 0; sequence < 2; ++sequence) {
    require(gst_app_src_push_buffer(GST_APP_SRC(source), gst_buffer_copy(input_template.get())) ==
                GST_FLOW_OK,
            "host async input push");
    if (!sequence)
      wait_until([&] { return core->outputs_ready.load() == 1; }, "host async first queue");
  }
  wait_until([&] { return core->pipeline.stream.stats().pull_count >= 2; },
             "host async second decode was not observed");
  require(core->outputs_ready.load() == 1 && core->outputs_dropped.load() == 0 &&
              !core->pipeline.copy_output_latched.load(),
          "host async pressure changed Block/copy policy");
  neat::Sample first, second;
  require(core->pull(5000, first) == neat::PullStatus::Ok, "host async first pull");
  require(core->pull(5000, second) == neat::PullStatus::Ok, "host async second pull");
  require(internal::sample_has_certified_terminal_host_memory(first) &&
              internal::sample_has_certified_terminal_host_memory(second) &&
              internal::sample_has_zero_copy_loans(first),
          "host async selected holder/loan");
  core->close();
  oracle(first.tensors.front());
  oracle(second.tensors.front());
  // Public tensor plus an open mapping keep the exact generation alive after close.
  auto mapping = tensor.storage->map(neat::MapMode::Read);
  require(gst_element_set_state(pipeline.get(), GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "terminal host pipeline close");
  raw.reset();
  require(internal::tensor_has_certified_terminal_host_memory(tensor), "close revoked live holder");
  oracle(tensor);
  mixed = {};
  require(gate->inflight() > 0, "held aliases must retain the host loan");
  mapping = {};
  outside = {};
  ambiguous = {};
  output = {};
  foreign_output = {};
  require(gate->inflight() == 0 && gate->released() == static_cast<std::uint64_t>(acquired) &&
              gate->overrelease() == 0,
          "final host loan release was not exactly once");
}
