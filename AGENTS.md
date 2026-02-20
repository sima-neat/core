# Agent Guidance

This file defines baseline quality expectations for automated and human contributors.

## Required standards

- Follow the coding rules in `docs/contribute/coding_standard.md`.
- Follow the testing rules in `docs/contribute/test_requirements.md`.
- Treat public headers under `include/*` as stable API.

## API compatibility

- Prefer non-breaking API changes.
- Any breaking public API signature change must follow the documented process in `docs/contribute/coding_standard.md`.
- PRs with breaking API changes must complete the `Breaking API Change` section in `.github/pull_request_template.md`.

## Documentation updates

When behavior or public API changes:

- Update `docs/contribute/architecture.md` if architecture/contracts changed.
- Update user-facing docs for workflow/configuration changes.
- Update examples if API usage changed.

## Definition of done

A change is ready when:

- Relevant tests were added/updated and pass.
- Docs are updated for user-visible changes.
- API compatibility impact was assessed and documented.
