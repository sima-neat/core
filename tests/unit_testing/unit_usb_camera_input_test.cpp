// nodes::UsbCameraInput: option validation, stable USB camera
// IDs from format-faithful synthetic sysfs trees, and the generated pipeline. Camera
// access itself (mode listing, first-frame sampling) is covered on a DevKit.
#include "nodes/io/UsbCameraInput.h"
#include "nodes/io/internal/UvcCamera.h"
#include "pipeline/Graph.h"

#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
namespace uvc = simaai::neat::internal::uvc;
using simaai::neat::FormatTag;
using simaai::neat::Graph;
using simaai::neat::Node;
using simaai::neat::UsbCameraInfo;
using simaai::neat::UsbCameraInputOptions;
using simaai::neat::nodes::UsbCameraInput;

static_assert(std::is_same_v<decltype(UsbCameraInput(std::declval<UsbCameraInputOptions>())),
                             std::shared_ptr<Node>>);
static_assert(std::is_same_v<
              decltype(simaai::neat::usb_camera_info(std::declval<const UsbCameraInputOptions&>())),
              UsbCameraInfo>);
static_assert(std::is_same_v<decltype(UsbCameraInput(std::declval<UsbCameraInputOptions>(),
                                                     std::declval<const UsbCameraInfo&>())),
                             std::shared_ptr<Node>>);

namespace {

// A one-node Graph, so describe_backend() renders the camera's fragment.
Graph camera_graph(const UsbCameraInputOptions& opt, const uvc::ResolvedCamera& camera) {
  Graph graph;
  graph.add(uvc::camera_node(opt, camera));
  return graph;
}

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos)
    throw std::runtime_error(msg + " (unexpected: " + needle + ")");
}

template <typename E>
void require_throws(const std::function<void()>& fn, const std::string& needle,
                    const std::string& label) {
  try {
    fn();
  } catch (const E& e) {
    require_contains(e.what(), needle, label);
    return;
  }
  throw std::runtime_error(label + ": expected exception");
}

// A sysfs tree laid out like a Modalix DevKit's: xHCI on PCIe, the first
// firmware-described ancestor being the PCIe host bridge.
struct FakeSysfs {
  fs::path root;
  FakeSysfs() {
    char tmpl[] = "/tmp/uvc_sysfs_XXXXXX";
    root = fs::path(::mkdtemp(tmpl));
    fs::create_directories(root / "firmware/devicetree/base/pci@7000000000");
    fs::create_directories(root / "bus/usb/drivers/uvcvideo");
    fs::create_directories(root / "bus/platform/drivers/simaai-v4l2-vid");
    fs::create_directories(root / "class/video4linux");
    fs::create_directories(host());
    fs::create_directory_symlink("../../../firmware/devicetree/base/pci@7000000000",
                                 host() / "of_node");
  }
  ~FakeSysfs() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  fs::path host() const {
    return root / "devices/platform/7ff0000000.pci";
  }
  fs::path usb_device(const std::string& port) const {
    return host() / "pci0003:00/0003:00:00.0/0003:01:00.0/usb1/1-3" / ("1-" + port);
  }
  static void write(const fs::path& file, const std::string& text) {
    std::ofstream(file) << text << "\n";
  }
  // USB camera at `port` with interface 1.0; one node per entry of `nodes`, indexed in order.
  void camera(const std::string& port, const std::vector<std::string>& nodes,
              bool with_product = true) {
    const fs::path dev = usb_device(port);
    const fs::path iface = dev / ("1-" + port + ":1.0");
    fs::create_directories(iface);
    write(dev / "idVendor", "046d");
    if (with_product)
      write(dev / "idProduct", "08e5");
    fs::create_directory_symlink(fs::relative(root / "bus/usb/drivers/uvcvideo", iface),
                                 iface / "driver");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const fs::path node = root / "class/video4linux" / nodes[i];
      fs::create_directories(node);
      fs::create_directory_symlink(fs::relative(iface, node), node / "device");
      write(node / "index", std::to_string(i));
    }
  }
  // A MIPI capture node (not uvcvideo).
  void mipi(const std::string& node) {
    const fs::path dev = root / "devices/platform/csi2video@1";
    fs::create_directories(dev);
    fs::create_directory_symlink(fs::relative(root / "bus/platform/drivers/simaai-v4l2-vid", dev),
                                 dev / "driver");
    fs::create_directories(root / "class/video4linux" / node);
    fs::create_directory_symlink(fs::relative(dev, root / "class/video4linux" / node),
                                 root / "class/video4linux" / node / "device");
  }
};

UsbCameraInputOptions mjpeg_options() {
  UsbCameraInputOptions opt;
  opt.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5";
  opt.width = 1920;
  opt.height = 1080;
  opt.framerate_num = 30;
  opt.framerate_den = 1;
  return opt;
}

void check_validation() {
  auto opt = mjpeg_options();
  opt.camera_name.clear();
  require_throws<std::invalid_argument>([&] { (void)UsbCameraInput(opt); }, "requires camera_name",
                                        "missing camera ID");
  opt = mjpeg_options();
  opt.framerate_num = 0;
  require_throws<std::invalid_argument>([&] { (void)UsbCameraInput(opt); },
                                        "nonzero width, height and frame rate", "zero rate");
  opt = mjpeg_options();
  opt.buffer_name.clear();
  require_throws<std::invalid_argument>([&] { (void)UsbCameraInput(opt); },
                                        "requires a buffer_name", "empty buffer name");
  require_throws<std::invalid_argument>([&] { (void)simaai::neat::usb_camera_info(opt); },
                                        "requires a buffer_name", "usb_camera_info validates");
  opt = mjpeg_options();
  opt.queue_depth = 59;
  require_throws<std::invalid_argument>([&] { (void)UsbCameraInput(opt); },
                                        "queue_depth exceeds 58", "capture pool above 64 buffers");
}

void check_ids_sorted() {
  FakeSysfs sys;
  sys.camera("3.2", {"video1"}); // plugged first, so it has the lower device number
  sys.camera("3.1", {"video5"});
  const auto ids = uvc::camera_ids(uvc::list_nodes(sys.root.string()));
  require(ids.size() == 2 && ids[0] == "/base/pci@7000000000-3.1:1.0-046d:08e5" &&
              ids[1] == "/base/pci@7000000000-3.2:1.0-046d:08e5",
          "IDs are sorted by ID, not by plug order");
}

void check_reconnect() {
  FakeSysfs sys;
  sys.camera("3.1", {"video1"});
  const std::string id = "/base/pci@7000000000-3.1:1.0-046d:08e5";
  const uvc::ResolvedCamera camera{"/dev/video1", FormatTag::NV12, id, sys.root.string(), 0};
  const Graph graph = camera_graph(mjpeg_options(), camera);
  require_contains(graph.describe_backend(), "device=/dev/video1", "camera still at video1");
  // Replugged into port 3.2: video1 now belongs to a camera with another ID.
  fs::remove(sys.root / "class/video4linux/video1/device");
  sys.camera("3.2", {"video1"});
  require_throws<std::runtime_error>([&] { (void)graph.describe_backend(); },
                                     "is no longer /dev/video1", "moved camera is not reopened");
}

void check_renumbered_to_metadata_node() {
  FakeSysfs sys;
  sys.camera("3.1", {"video1", "video2"}); // capture + metadata node
  const std::string id = "/base/pci@7000000000-3.1:1.0-046d:08e5";
  const uvc::ResolvedCamera camera{"/dev/video1", FormatTag::NV12, id, sys.root.string(), 0};
  const Graph graph = camera_graph(mjpeg_options(), camera);
  require_contains(graph.describe_backend(), "device=/dev/video1", "capture node still at video1");
  // Re-enumerated: video1 is now the same camera's metadata node, so the ID still matches.
  FakeSysfs::write(sys.root / "class/video4linux/video1/index", "1");
  require_throws<std::runtime_error>([&] { (void)graph.describe_backend(); },
                                     "is no longer /dev/video1", "metadata node is not opened");
  // No index file: unknown, so the ID check alone decides.
  fs::remove(sys.root / "class/video4linux/video1/index");
  require_contains(graph.describe_backend(), "device=/dev/video1", "missing index is unknown");
}

void check_ids_and_nodes() {
  FakeSysfs sys;
  sys.camera("3.1", {"video1", "video2"}); // capture + metadata node, one interface
  sys.camera("3.2", {"video10"});          // an identical camera on another port
  sys.camera("3.3", {"video3"}, false);    // no idProduct: omitted, not guessed
  sys.mipi("video0");
  const auto nodes = uvc::list_nodes(sys.root.string());
  require(nodes.size() == 3, "expected three uvcvideo nodes, got " + std::to_string(nodes.size()));
  const std::string a = "/base/pci@7000000000-3.1:1.0-046d:08e5";
  const std::string b = "/base/pci@7000000000-3.2:1.0-046d:08e5";
  require(nodes[0].device == "/dev/video1" && nodes[0].id == a, "video1 id: " + nodes[0].id);
  require(nodes[1].device == "/dev/video2" && nodes[1].id == a, "metadata node shares the id");
  require(nodes[2].device == "/dev/video10" && nodes[2].id == b,
          "identical camera on another port has its own id, sorted after video2");
  // The libcamera format: no bus number, so the ID survives USB bus renumbering.
  require_not_contains(a, "1-3.1", "bus number must not be part of the id");

  require_throws<std::runtime_error>(
      [&] { (void)uvc::resolve_mjpeg_camera("missing", 1920, 1080, 30, 1, sys.root.string()); },
      "connected USB camera IDs: " + a + ", " + b, "unknown id lists the connected ids");
}

void check_acpi_controller() {
  FakeSysfs sys;
  fs::remove(sys.host() / "of_node");
  fs::create_directories(sys.host() / "firmware_node");
  FakeSysfs::write(sys.host() / "firmware_node/path", "\\_SB_.PCI0.XHC_");
  sys.camera("3.1", {"video1"});
  const auto nodes = uvc::list_nodes(sys.root.string());
  require(nodes.size() == 1 && nodes[0].id == "\\_SB_.PCI0.XHC_-3.1:1.0-046d:08e5",
          "ACPI controller id: " + (nodes.empty() ? std::string("none") : nodes[0].id));
}

void check_no_cameras() {
  FakeSysfs sys;
  sys.mipi("video0");
  require(uvc::list_nodes(sys.root.string()).empty(), "a MIPI node is not a USB camera");
  require_throws<std::runtime_error>(
      [&] { (void)uvc::resolve_mjpeg_camera("x", 640, 480, 30, 1, sys.root.string()); },
      "no USB cameras are connected", "empty system");
}

void check_jpeg_pipeline() {
  auto opt = mjpeg_options();
  opt.framerate_num = 30000;
  opt.framerate_den = 1001;
  const auto node = uvc::camera_node(opt, {"/dev/video1", FormatTag::NV16});
  require(node->kind() == "UsbCameraInput", "a live source kind");
  require(node->is_live_source(), "the camera is a live source");
  const std::string text = node->backend_fragment(2);
  require(text == "v4l2src name=n2_camera_src device=/dev/video1 io-mode=dmabuf-import ! "
                  "capsfilter name=n2_camera_caps caps=\"image/jpeg,width=1920,height=1080,"
                  "framerate=30000/1001\" ! neatcapturepool name=n2_camera_pool buffers=8 ! "
                  "identity name=n2_camera_drop drop-buffer-flags=corrupted ! queue "
                  "name=n2_camera_queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 "
                  "leaky=downstream",
          "Jpeg fragment: " + text);
  require_not_contains(text, "neatdecoder", "the node has no decoder");
  require(node->element_names(2) == std::vector<std::string>{"n2_camera_src", "n2_camera_caps",
                                                             "n2_camera_pool", "n2_camera_drop",
                                                             "n2_camera_queue"},
          "Jpeg element names match the fragment");
  const auto* provider = dynamic_cast<const simaai::neat::OutputSpecProvider*>(node.get());
  require(provider != nullptr, "Jpeg mode publishes an output spec");
  const auto spec = provider->output_spec({});
  require(spec.payload_type == simaai::neat::PayloadType::Encoded &&
              spec.media_type == "image/jpeg" && spec.format == "JPEG" && spec.width == 1920 &&
              spec.height == 1080 && spec.fps_num == 30000 && spec.fps_den == 1001,
          "Jpeg output spec describes encoded JPEG frames");

  // The queue follows queue_depth, and an exact fractional mode stays exact in the caps.
  opt.queue_depth = 3;
  opt.framerate_num = 100;
  opt.framerate_den = 3;
  const std::string tuned = camera_graph(opt, {"/dev/video1", FormatTag::NV12}).describe_backend();
  require_contains(tuned, "max-size-buffers=3", "queue depth");
  require_contains(tuned, "buffers=9",
                   "the capture pool covers the queue and the frames held downstream");
  require_contains(tuned, "framerate=100/3", "fractional mode caps");

  opt.insert_queue = false;
  const auto bare = uvc::camera_node(opt, {"/dev/video1", FormatTag::NV12});
  require_not_contains(bare->backend_fragment(0), "queue", "no queue when insert_queue is off");
  require(bare->element_names(0).size() == 4, "no queue name when insert_queue is off");
  require_contains(bare->backend_fragment(0), "buffers=6", "no queue: only the downstream hold");
}

void check_jpeg_reconnect() {
  FakeSysfs sys;
  sys.camera("3.1", {"video1"});
  auto opt = mjpeg_options();
  const uvc::ResolvedCamera camera{"/dev/video1", FormatTag::NV12, opt.camera_name,
                                   sys.root.string(), 0};
  const auto node = uvc::camera_node(opt, camera);
  require_contains(node->backend_fragment(0), "device=/dev/video1", "Jpeg camera at video1");
  fs::remove(sys.root / "class/video4linux/video1/device");
  sys.camera("3.2", {"video1"});
  require_throws<std::runtime_error>([&] { (void)node->backend_fragment(0); },
                                     "is no longer /dev/video1", "reconnects are checked");
}

// UsbCameraInput(opt, info) reuses usb_camera_info()'s result: sysfs only, no device open.
void check_from_info() {
  FakeSysfs sys;
  sys.camera("3.1", {"video1", "video2"}); // capture + metadata node
  const auto opt = mjpeg_options();
  // No such file: opening it would fail, so a match shows the device is not touched.
  const std::string device = (sys.root / "dev/video1").string();
  const UsbCameraInfo info{device, FormatTag::NV16, 0};
  const auto from_info = uvc::camera_from_info(opt.camera_name, info, sys.root.string());
  // What resolve_mjpeg_camera() returns for the same camera, after it read a frame.
  const uvc::ResolvedCamera probed{device, FormatTag::NV16, opt.camera_name, sys.root.string(), 0};
  const std::string fragment = uvc::camera_node(opt, from_info)->backend_fragment(0);
  require(fragment == uvc::camera_node(opt, probed)->backend_fragment(0),
          "info-based node matches the probed one: " + fragment);
  require_contains(camera_graph(opt, from_info).describe_backend(), "device=" + device,
                   "info-based node builds");
  require(!fs::exists(device), "the device was never created or opened");

  // Options are validated before the info is looked at.
  auto bad = opt;
  bad.buffer_name.clear();
  require_throws<std::invalid_argument>([&] { (void)UsbCameraInput(bad, info); },
                                        "requires a buffer_name", "info overload validates");

  const auto stale = [&](const UsbCameraInfo& i, const std::string& label) {
    require_throws<std::runtime_error>(
        [&] { (void)uvc::camera_from_info(opt.camera_name, i, sys.root.string()); },
        "Call usb_camera_info again", label);
  };
  // Info for the metadata node, or renumbered: the capture node index no longer matches.
  stale({(sys.root / "dev/video2").string(), FormatTag::NV16, 0}, "metadata node is not opened");
  // Another camera's info: video10 belongs to port 3.2.
  sys.camera("3.2", {"video10"});
  stale({(sys.root / "dev/video10").string(), FormatTag::NV16, 0}, "info of another camera");
  // Replugged into port 3.3: video1 now belongs to a camera with another ID.
  fs::remove(sys.root / "class/video4linux/video1/device");
  sys.camera("3.3", {"video1"});
  stale(info, "moved camera");
  // Unplugged: the node is gone.
  fs::remove_all(sys.root / "class/video4linux/video1");
  stale(info, "unplugged camera");
}

} // namespace

int main() {
  try {
    check_validation();
    check_ids_sorted();
    check_reconnect();
    check_renumbered_to_metadata_node();
    check_ids_and_nodes();
    check_acpi_controller();
    check_no_cameras();
    check_jpeg_pipeline();
    check_jpeg_reconnect();
    check_from_info();
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  std::cout << "[OK] unit_usb_camera_input_test\n";
  return 0;
}
