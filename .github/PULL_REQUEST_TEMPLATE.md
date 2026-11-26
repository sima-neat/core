## Summary

<!-- Link the issue using Fixes #<issue> or Refs #<issue>. Describe what changed and why. -->

## Risk

<!-- Low, Medium, or High. Explain the affected behavior and scope. -->

## Test evidence

<!-- Link the passing test result or give the command and outcome. For a bug fix, describe the original failure and the regression that now passes. For device behavior, name the target and package tested. List any outstanding checks. -->

## API, ABI, and migration

<!-- Write None if unaffected. Describe changes to C++ or PyNeat behavior, signatures, layout, or exported symbols. For a break, identify the affected symbols and users, migration steps, release intent, and maintainer approval. Compare abi-version with main using the release checklist. -->

## Docs impact

<!-- Link any guide, example, or skill updates. Include required translations and source hashes in this PR. Use No docs update needed if existing guidance still applies. -->

## Author checks

- [ ] Each new option, helper, or fallback serves a concrete case. I removed unrelated changes.
- [ ] I trimmed comments that repeat the code and limited documentation edits to guidance readers need.
- [ ] Before marking this PR ready for review, I reviewed the full diff against `develop` using `core-review-pr` and addressed the applicable findings.

**Review outcome** <!-- Required. One sentence describing what you fixed during review, or No findings. -->
