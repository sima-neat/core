"""Pure classifiers for connected-card GenAI admission and fault evidence."""

import re


def _normalized(message):
  return re.sub(r"\s+", " ", str(message).casefold()).strip()


def is_bounded_competing_session_rejection(message):
  text = _normalized(message)
  concepts = (
      "admission rejected",
      "another session",
      "busy",
      "capacity",
      "concurrent session",
      "resource unavailable",
      "resource exhausted",
      "session limit",
  )
  return any(concept in text for concept in concepts)


def is_injected_transport_failure(message):
  text = _normalized(message)
  concepts = (
      "connection reset",
      "genai worker stopped responding",
      "pcie service notify failed",
      "pcie service receive failed",
      "transport closed",
  )
  return any(concept in text for concept in concepts)
