import json
import socket

import pyneat
import pytest


def _udp_receiver():
  rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  rx.bind(("127.0.0.1", 0))
  rx.settimeout(2.0)
  return rx


def test_optiview_json_helper_and_sender_loopback():
  rx = _udp_receiver()
  try:
    json_port = rx.getsockname()[1]

    channel = pyneat.OptiViewChannelOptions()
    channel.host = "127.0.0.1"
    channel.channel = 2
    channel.video_port_base = 9000
    channel.json_port_base = json_port - 2

    sender = pyneat.OptiViewJsonOutput(channel)

    assert sender.ok() is True
    assert sender.host() == "127.0.0.1"
    assert sender.video_port() == 9002
    assert sender.json_port() == json_port

    obj = pyneat.OptiViewObject()
    obj.x = 10
    obj.y = 20
    obj.w = 30
    obj.h = 40
    obj.score = 0.95
    obj.class_id = 0

    payload = pyneat.OptiViewMakeJson(12345, "frame-7", [obj], ["person"])
    parsed = json.loads(payload)
    assert parsed["type"] == "object-detection"
    assert parsed["timestamp"] == 12345
    assert parsed["frame_id"] == "frame-7"
    assert parsed["data"]["objects"][0]["label"] == "person"
    assert parsed["data"]["objects"][0]["bbox"] == [10, 20, 30, 40]

    assert sender.send_json(payload) is True
    received, _ = rx.recvfrom(4096)
    assert received.decode("utf-8") == payload

    assert sender.send_detection(222, "frame-8", [obj], ["person"]) is True
    received, _ = rx.recvfrom(4096)
    parsed_detection = json.loads(received.decode("utf-8"))
    assert parsed_detection["timestamp"] == 222
    assert parsed_detection["frame_id"] == "frame-8"
    assert parsed_detection["data"]["objects"][0]["confidence"] == pytest.approx(0.95)
  finally:
    rx.close()
