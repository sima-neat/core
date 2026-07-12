import json
import socket

import pyneat


def _udp_receiver():
  rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  rx.bind(("127.0.0.1", 0))
  rx.settimeout(2.0)
  return rx


def test_metadata_sender_loopback():
  rx = _udp_receiver()
  try:
    metadata_port = rx.getsockname()[1]

    channel = pyneat.MetadataSenderOptions()
    channel.host = "127.0.0.1"
    channel.channel = 2
    channel.metadata_port_base = metadata_port - 2
    send_options = pyneat.MetadataSenderSendOptions()
    send_options.nonblocking = True

    sender = pyneat.MetadataSender(channel, send_options)

    assert sender.ok() is True
    assert sender.host() == "127.0.0.1"
    assert sender.metadata_port() == metadata_port
    assert sender.nonblocking() is True

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

    stats = sender.stats()
    assert stats.send_attempts == 2
    assert stats.datagrams_sent == 2
    assert stats.send_failures == 0
    assert stats.would_block == 0
    assert stats.no_buffer_space == 0
    assert stats.last_errno == 0
    assert stats.max_send_duration_ns >= stats.last_send_duration_ns
  finally:
    rx.close()
