#!/usr/bin/env python3
"""Fail-closed release gate for a Windows PCIe-host cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windows_candidate import validate_candidate


def read_object(path: Path, description: str) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid {description}: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{description} must be a JSON object")
  return value


def verify(candidate_dir: Path, qualification_path: Path, expected_commit: str,
           require_signatures: bool, require_genai: bool = False) -> dict:
  candidate = validate_candidate(candidate_dir)
  if candidate.core_commit != expected_commit:
    raise ValueError(
        f"candidate Core commit {candidate.core_commit} is not workflow commit {expected_commit}")

  qualification = read_object(qualification_path, "hardware qualification")
  if qualification.get("result") != "PASS":
    raise ValueError("hardware qualification did not pass")
  qualified_candidate = qualification.get("candidate")
  if not isinstance(qualified_candidate, dict):
    raise ValueError("hardware qualification has no candidate identity")
  if qualified_candidate.get("core_commit") != candidate.core_commit:
    raise ValueError("hardware qualification covered a different Core commit")
  if qualified_candidate.get("resources") != candidate.resources:
    raise ValueError("hardware qualification covered different package bytes")
  cleanup = qualification.get("cleanup")
  if not isinstance(cleanup, dict) or cleanup.get("errors") != []:
    raise ValueError("hardware qualification did not prove clean removal")
  final_card = qualification.get("card_final")
  if (not isinstance(final_card, dict) or final_card.get("workers") != "0"
      or final_card.get("genai_workers") != "0"):
    raise ValueError("hardware qualification left a worker on the card")
  tests = qualification.get("tests")
  if (not isinstance(tests, dict) or not tests.get("cpp") or not tests.get("python")):
    raise ValueError("hardware qualification did not run both customer APIs")
  if require_genai:
    names = set(tests["cpp"]) | set(tests["python"])
    required_genai = {
        "test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_vlm_run_and_stream",
        "test_genai_asr_file_pcm_translation_and_stream",
    }
    if not required_genai.issubset(names):
      raise ValueError("release qualification did not run the complete Windows GenAI matrix")

  signatures = read_object(candidate.root / "windows-signatures.json", "signature receipt")
  records = signatures.get("records")
  if not isinstance(records, list) or not records:
    raise ValueError("signature receipt has no records")
  invalid = [record.get("path") for record in records
             if not isinstance(record, dict) or record.get("status") != "Valid"]
  if require_signatures:
    if not signatures.get("release_required"):
      raise ValueError("candidate was not built in signed release mode")
    if invalid:
      raise ValueError(f"release candidate has invalid Authenticode records: {invalid}")
    if not signatures.get("certificate_sha1") or not signatures.get("timestamp_url"):
      raise ValueError("signed release receipt has no certificate or timestamp authority")
    expected_thumbprint = str(signatures["certificate_sha1"]).upper()
    wrong_publishers = [
        record.get("path") for record in records
        if str(record.get("thumbprint", "")).upper() != expected_thumbprint
    ]
    if wrong_publishers:
      raise ValueError(f"release candidate has an unexpected signer: {wrong_publishers}")
    missing_timestamps = [
        record.get("path") for record in records if not record.get("timestamp_thumbprint")
    ]
    if missing_timestamps:
      raise ValueError(f"release candidate has no trusted timestamp: {missing_timestamps}")

  return {
      "core_commit": candidate.core_commit,
      "hardware_qualified": True,
      "genai_qualified": require_genai,
      "resources": candidate.resources,
      "signatures_required": require_signatures,
      "signatures_valid": not invalid,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--qualification", type=Path, required=True)
  parser.add_argument("--require-genai", action="store_true")
  parser.add_argument("--require-signatures", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = verify(
      args.candidate, args.qualification, args.expected_commit,
      args.require_signatures, args.require_genai)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
