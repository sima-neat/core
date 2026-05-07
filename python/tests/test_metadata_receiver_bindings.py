import json
import socket

import pyneat
import pytest


def _udp_receiver():
  rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  rx.bind(("127.0.0.1", 0))
  rx.settimeout(2.0)
  return rx


def test_metadata_receiver_helper_and_sender_loopback():
  rx = _udp_receiver()
  try:
    metadata_port = rx.getsockname()[1]

    channel = pyneat.MetadataReceiverChannelOptions()
    channel.host = "127.0.0.1"
    channel.channel = 2
    channel.metadata_port_base = metadata_port - 2

    sender = pyneat.MetadataReceiverOutput(channel)

    assert sender.ok() is True
    assert sender.host() == "127.0.0.1"
    assert sender.metadata_port() == metadata_port

    generic = pyneat.MetadataReceiverPayload()
    generic.type = "tracking"
    generic.data_json = '{"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]}'
    generic.timestamp_ms = 12345
    generic.frame_id = "frame-7"

    generic_payload = pyneat.MetadataReceiverMakeJson(generic)
    parsed = json.loads(generic_payload)
    assert parsed["type"] == "tracking"
    assert parsed["timestamp"] == 12345
    assert parsed["frame_id"] == "frame-7"
    assert parsed["data"]["tracks"][0]["id"] == "trk-1"

    assert sender.send_metadata(generic) is True
    received, _ = rx.recvfrom(4096)
    assert received.decode("utf-8") == generic_payload

    obj = pyneat.MetadataReceiverObject()
    obj.x = 10
    obj.y = 20
    obj.w = 30
    obj.h = 40
    obj.score = 0.95
    obj.class_id = 0

    detection_payload = pyneat.MetadataReceiverMakeObjectDetectionJson(
        222, "frame-8", [obj], ["person"]
    )
    parsed_detection = json.loads(detection_payload)
    assert parsed_detection["type"] == "object-detection"
    assert parsed_detection["timestamp"] == 222
    assert parsed_detection["frame_id"] == "frame-8"
    assert parsed_detection["data"]["objects"][0]["label"] == "person"
    assert parsed_detection["data"]["objects"][0]["bbox"] == [10, 20, 30, 40]

    assert sender.send_object_detection(333, "frame-9", [obj], ["person"]) is True
    received, _ = rx.recvfrom(4096)
    parsed_sent_detection = json.loads(received.decode("utf-8"))
    assert parsed_sent_detection["timestamp"] == 333
    assert parsed_sent_detection["frame_id"] == "frame-9"
    assert parsed_sent_detection["data"]["objects"][0]["confidence"] == pytest.approx(0.95)
  finally:
    rx.close()
