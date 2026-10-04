#include "HostPcieOutputLayout.h"
#include "HostPcieTensorSetMeta.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace simaai::neat::pcie::internal;

namespace {
void check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

GstCaps* caps(std::uint64_t raw, std::initializer_list<std::uint64_t> pairs) {
  auto* c = gst_caps_new_empty_simple("application/vnd.simaai.tensor");
  auto* s = gst_caps_get_structure(c, 0);
  gst_structure_set(s, "pcie-return-layout-version", G_TYPE_UINT, 2u, "pcie-return-raw-bytes",
                    G_TYPE_UINT64, static_cast<guint64>(raw), nullptr);
  GValue array = G_VALUE_INIT;
  GValue value = G_VALUE_INIT;
  g_value_init(&array, GST_TYPE_ARRAY);
  g_value_init(&value, G_TYPE_UINT64);
  for (auto n : pairs) {
    g_value_set_uint64(&value, n);
    gst_value_array_append_value(&array, &value);
  }
  gst_structure_set_value(s, "pcie-return-regions", &array);
  g_value_unset(&value);
  g_value_unset(&array);
  return c;
}

void rejects(const PcieModelFacts& facts, GstCaps* c) {
  bool rejected = false;
  try {
    (void)resolve_output_layout(facts, c);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  gst_caps_unref(c);
  check(rejected, "invalid layout accepted");
}
} // namespace

int main() {
  gst_init(nullptr, nullptr);
  try {
    PcieModelFacts facts;
    facts.packed_output_bytes = 8;
    PcieTensorFact first;
    first.name = "first";
    first.dtype = "UINT8";
    first.shape = {4};
    first.size_bytes = 4;
    auto second = first;
    second.name = "second";
    second.payload_offset = 4;
    facts.outputs = {first, second};
    auto* c = caps(16, {12, 4, 0, 4});
    auto layout = resolve_output_layout(facts, c);
    check(layout.raw_bytes == 16, "raw extent lost");
    check(layout.facts.outputs[0].payload_offset == 12 &&
              layout.facts.outputs[1].payload_offset == 0,
          "physical order lost");
    auto owner = std::make_shared<MappedSample>();
    auto* buffer = gst_buffer_new_allocate(nullptr, 16, nullptr);
    const char bytes[] = "BBBB--------AAAA";
    gst_buffer_fill(buffer, 0, bytes, 16);
    owner->sample = gst_sample_new(buffer, c, nullptr, nullptr);
    owner->buffer = buffer;
    check(gst_buffer_map(buffer, &owner->map, GST_MAP_READ), "map failed");
    owner->mapped = true;
    gst_buffer_unref(buffer);
    gst_caps_unref(c);
    auto tensors = tensors_from_output_payload(owner, layout.facts);
    check(tensors[0].data == owner->map.data + 12 && tensors[1].data == owner->map.data,
          "outputs must be views without a repacking copy");
    check(tensors[0].route.name == "first" && tensors[1].route.name == "second",
          "same-sized outputs confused");
    std::weak_ptr<MappedSample> weak = owner;
    owner.reset();
    check(!weak.expired() && std::memcmp(tensors[0].data, "AAAA", 4) == 0 &&
              std::memcmp(tensors[1].data, "BBBB", 4) == 0,
          "view owner lost");
    tensors.clear();
    check(weak.expired(), "output owner leaked");

    rejects(facts, caps(16, {0, 4, 2, 4}));
    rejects(facts, caps(16, {15, 4, 0, 4}));
    rejects(facts, caps(16, {12, 3, 0, 4}));
    rejects(facts, caps(16, {12, 4, 0}));
    rejects(facts, caps(16, {std::numeric_limits<std::uint64_t>::max(), 4, 0, 4}));
    auto* bad_version = caps(16, {12, 4, 0, 4});
    gst_structure_set(gst_caps_get_structure(bad_version, 0), "pcie-return-layout-version",
                      G_TYPE_UINT, 99u, nullptr);
    rejects(facts, bad_version);
    auto* legacy = gst_caps_new_empty_simple("application/vnd.simaai.tensor");
    check(resolve_output_layout(facts, legacy).facts.outputs[1].payload_offset == 4,
          "legacy packed layout changed");
    gst_caps_unref(legacy);

    // MLA-only logical heads can be strided views within one packed carrier.
    facts.packed_output_bytes = 8;
    first.shape = {2, 2};
    first.transport_strides_bytes = {4, 1};
    first.payload_offset = 1;
    facts.outputs = {first};
    auto* padded = caps(24, {16, 8});
    facts.dense_output_bytes = 4;
    auto padded_layout = resolve_output_layout(facts, padded);
    check(padded_layout.facts.outputs[0].payload_offset == 17, "intra-carrier offset lost");
    auto padded_owner = std::make_shared<MappedSample>();
    auto* padded_buffer = gst_buffer_new_allocate(nullptr, 24, nullptr);
    gst_buffer_fill(padded_buffer, 16, "-AB--CD-", 8);
    padded_owner->sample = gst_sample_new(padded_buffer, padded, nullptr, nullptr);
    padded_owner->buffer = padded_buffer;
    check(gst_buffer_map(padded_buffer, &padded_owner->map, GST_MAP_READ), "padded map failed");
    padded_owner->mapped = true;
    gst_buffer_unref(padded_buffer);
    auto compacted = tensors_from_output_payload(padded_owner, padded_layout.facts);
    std::weak_ptr<MappedSample> padded_weak = padded_owner;
    padded_owner.reset();
    check(padded_weak.expired() && std::memcmp(compacted[0].data, "ABCD", 4) == 0,
          "MLA-only compaction must use the remapped offset and release the raw owner");
    check(compacted[0].strides_bytes == std::vector<std::int64_t>({2, 1}),
          "compacted output is not contiguous");
    first.payload_offset = 3; // Six-byte strided span would cross the carrier.
    facts.outputs = {first};
    rejects(facts, padded);
    std::cout << "PCIe output layout tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
