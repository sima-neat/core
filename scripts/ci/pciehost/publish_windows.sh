#!/usr/bin/env bash
# Publish one verified Windows PCIe-host cohort to its Vulcan layout:
# core/<ref>/<commit12>/pciehost/windows/amd64/, as for the Ubuntu host packages.
# Usage: publish_windows.sh <candidate-dir> <verification-dir>
# Environment: ARTIFACT_BUCKET, REF_TYPE (branch|tag), REF_NAME, GITHUB_SHA,
# optional SSE_KMS_KEY_ID and CLOUDFRONT_DISTRIBUTION_ID.
set -euo pipefail
candidate="$1"
verification="$2"
ref_args=(--bucket "${ARTIFACT_BUCKET}" --ref-type "${REF_TYPE}" --ref "${REF_NAME}"
  --commit "${GITHUB_SHA}" --candidate "${candidate}")
plan="$(python3 scripts/ci/pciehost/windows_release.py publication "${ref_args[@]}")"
mapfile -t paths <<< "${plan}"
prefix="${paths[0]}"
if (( ${#paths[@]} == 1 )); then
  echo "s3://${ARTIFACT_BUCKET}/${prefix}/ already holds this release; nothing to publish"
  exit 0
fi
s3_args=(--sse aws:kms)
if [[ -n "${SSE_KMS_KEY_ID:-}" ]]; then
  s3_args+=(--sse-kms-key-id "${SSE_KMS_KEY_ID}")
fi
for path in "${paths[@]:1}"; do
  name="$(basename "${path}")"
  # Metadata becomes visible only after the uploaded files are verified.
  case "${name}" in metadata.json) name="${name}.pending" ;; esac
  case "${name}" in
    *.msi) content_type="application/x-msi" ;;
    *.sha256) content_type="text/plain" ;;
    *.json.pending) content_type="application/json" ;;
    *) content_type="application/octet-stream" ;;
  esac
  aws s3 cp "${path}" "s3://${ARTIFACT_BUCKET}/${prefix}/${name}" \
    --content-type "${content_type}" "${s3_args[@]}"
done
python3 scripts/ci/pciehost/windows_release.py verify-upload "${ref_args[@]}" \
  --output "${verification}"
aws s3 cp "${candidate}/metadata.json" "s3://${ARTIFACT_BUCKET}/${prefix}/metadata.json" \
  --content-type application/json "${s3_args[@]}"
if [[ -n "${CLOUDFRONT_DISTRIBUTION_ID:-}" ]]; then
  aws cloudfront create-invalidation \
    --distribution-id "${CLOUDFRONT_DISTRIBUTION_ID}" --paths "/${prefix}/*"
fi
echo "Published s3://${ARTIFACT_BUCKET}/${prefix}/"
