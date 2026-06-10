#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XML_DIR="${ROOT}/docs/doxygen/out/xml"
OUT_DIR="${ROOT}/docs/reference/cppapi"
WEBSITE_API_DIR="${ROOT}/website/docs/reference/cppapi"
LEGACY_DIR="${ROOT}/docs/reference/api"
DOXYFILE_XML="${XML_DIR}/Doxyfile.xml"
DOXYFILE_SRC="${ROOT}/docs/doxygen/Doxyfile"

if [[ ! -d "${XML_DIR}" ]]; then
  echo "Doxygen XML not found at ${XML_DIR}."
  echo "Run doxygen first (docs/doxygen/Doxyfile) and try again."
  exit 1
fi

CMD="${DOXYGEN2DOCUSAURUS_CMD:-npx --yes --package @xpack/doxygen2docusaurus@2.0.0 --package fast-xml-parser@5.2.5 doxygen2docusaurus}"
read -r -a cmd_parts <<< "${CMD}"
if ! command -v "${cmd_parts[0]}" >/dev/null 2>&1; then
  echo "Missing command: ${cmd_parts[0]}"
  echo "Install Node.js/npm (for npx) or set DOXYGEN2DOCUSAURUS_CMD."
  echo "Example: DOXYGEN2DOCUSAURUS_CMD=\"npx --yes --package @xpack/doxygen2docusaurus@2.0.0 --package fast-xml-parser@5.2.5 doxygen2docusaurus\""
  exit 1
fi

mkdir -p "${OUT_DIR}"
"${ROOT}/tools/strip_empty_programlisting.py"
if [[ ! -f "${DOXYFILE_XML}" ]]; then
  project_name="$(sed -n 's/^PROJECT_NAME[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "${DOXYFILE_SRC}" | head -n1)"
  project_brief="$(sed -n 's/^PROJECT_BRIEF[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "${DOXYFILE_SRC}" | head -n1)"
  doxygen_version="$(doxygen --version 2>/dev/null || echo unknown)"

  # doxygen2docusaurus@2.0.0 expects Doxyfile.xml, but Doxygen 1.9.x may not emit it.
  cat > "${DOXYFILE_XML}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<doxyfile version="${doxygen_version}" xml:lang="en-US">
  <option id="PROJECT_NAME" default="no" type="string"><value>"${project_name}"</value></option>
  <option id="PROJECT_BRIEF" default="no" type="string"><value>"${project_brief}"</value></option>
</doxyfile>
EOF
fi

(
  cd "${ROOT}/website"
  "${cmd_parts[@]}" -C .
)

if [[ -d "${WEBSITE_API_DIR}" ]]; then
  rm -rf "${OUT_DIR}"
  mkdir -p "$(dirname "${OUT_DIR}")"
  mv "${WEBSITE_API_DIR}" "${OUT_DIR}"
  rmdir "${ROOT}/website/docs/reference" 2>/dev/null || true
  rmdir "${ROOT}/website/docs" 2>/dev/null || true
fi

if [[ -d "${LEGACY_DIR}" ]]; then
  rm -rf "${LEGACY_DIR}"
fi

rm -rf "${OUT_DIR}/indices"
rm -rf "${OUT_DIR}/pages"

cat > "${OUT_DIR}/_category_.json" <<'EOF'
{
  "label": "C++ Reference",
  "position": 2
}
EOF

python3 "${ROOT}/tools/postprocess_d2d_links.py"
python3 "${ROOT}/tools/postprocess_api_availability_tags.py" --docs-dir "${OUT_DIR}"

echo "API docs generated at ${OUT_DIR}"
