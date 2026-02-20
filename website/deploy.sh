#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"

AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-pipeline-session-doc}"
CLOUDFRONT_DISTRIBUTION_ID="${CLOUDFRONT_DISTRIBUTION_ID:-E27E3TBO4YSCCW}"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "Build output not found at ${BUILD_DIR}."
  echo "Run the docs build first (e.g. ./build.sh --doc)."
  exit 1
fi

if [[ -z "${S3_BUCKET}" ]]; then
  echo "Missing S3_BUCKET."
  exit 1
fi

echo "Deploying ${BUILD_DIR} to s3://${S3_BUCKET}/"
aws s3 sync "${BUILD_DIR}/" "s3://${S3_BUCKET}/" --delete --region "${AWS_REGION}"

if [[ -n "${CLOUDFRONT_DISTRIBUTION_ID}" ]]; then
  echo "Invalidating CloudFront distribution ${CLOUDFRONT_DISTRIBUTION_ID}"
  aws cloudfront create-invalidation \
    --distribution-id "${CLOUDFRONT_DISTRIBUTION_ID}" \
    --paths "/*" \
    --region "${AWS_REGION}"
else
  echo "CLOUDFRONT_DISTRIBUTION_ID not set; skipping invalidation."
fi
