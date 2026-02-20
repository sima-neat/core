# C++20 Migration Notes

Project standard is C++20.

Guidance for generated code:

- Set:
  - `CMAKE_CXX_STANDARD 20`
  - `CMAKE_CXX_STANDARD_REQUIRED ON`
  - `CMAKE_CXX_EXTENSIONS OFF`
- Avoid assumptions tied to older compiler behavior for warning handling.
- Keep code warning-clean under strict warning profiles used by this project.

If integrating legacy snippets:

- Replace deprecated style patterns with explicit types/initialization.
- Ensure designated initializers cover required fields where used.
