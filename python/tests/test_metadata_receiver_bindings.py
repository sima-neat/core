import json
import socket

import pyneat


def _udp_receiver():
  rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  rx.bind(("127.0.0.1", 0))
  rx.settimeout(2.0)
  return rx


def test_metadata_receiver_sender_loopback():
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

    assert sender.send_raw_json('{"type":"raw","data":{"ok":true}}') is True
    received, _ = rx.recvfrom(4096)
    assert json.loads(received.decode("utf-8"))["type"] == "raw"

    assert (
        sender.send_metadata(
            "tracking",
            '{"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]}',
            12345,
            "frame-7",
        )
        is True
    )

    received, _ = rx.recvfrom(4096)
    parsed = json.loads(received.decode("utf-8"))
    assert parsed["type"] == "tracking"
    assert parsed["timestamp"] == 12345
    assert parsed["frame_id"] == "frame-7"
    assert parsed["data"]["tracks"][0]["id"] == "trk-1"
  finally:
    rx.close()
