---
title: "リリースチェックリスト"
description: "リリースゲートポリシーと再現可能なリリース手順"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# リリースチェックリスト

このドキュメントは、正式なリリースゲートポリシーです。

## リリースをブロックする条件

以下のすべての条件が満たされている場合にのみ、リリースが許可されます。

1. トラッキングされているソース/ドキュメントファイルの変更競合マーカーがないこと。
2. 公開ドキュメントに対する命名規則のチェックに合格すること。
3. 構成/ビルドの整合性チェックに合格すること（`cmake -S . -B ...` および `cmake --build ... --target sima_neat`）。
4. ドキュメントのリンクチェックが厳格なモードで合格すること（`DOCS_STRICT_LINKS=1`）。
5. 生成ステップの後に、作業ツリーがクリーンな状態であること。
6. リリース前に、未解決のクラッシュ/正当性に関する問題がゼロであること、およびリリースリファレンス上でもゼロであること。
7. クラッシュ/正当性/ストレステスト/サニタイザーゲートが、リリースリファレンス上ですべて緑色であること。
8. モデルアーカイブセキュリティゲートが緑色であること（`model-archive-security-gate`）。
9. インストールスモークゲートが緑色であること（`install-smoke`）。
10. パフォーマンス回帰ゲートが緑色であること（`perf-regression-gate`）。
11. リリースタグに対するソーク安定性テストが緑色であること（`soak-weekly`）。
12. リリース候補に対するファズテストが緑色であること（`fuzz-nightly`）。
13. 厳格なテストレーンに対して、ゼロスキップゲートが緑色であること（`zero-skip-gate`）。
14. 必要なガバナンスファイルが存在し、有効であること：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
15. リリースメタデータが完了していること：
   - `project(SimaNeat VERSION x.y.z)` が `CMakeLists.txt` に更新されていること
   - 必要に応じて、`package-version` と `platform-version` が `deps/manifest.json` に更新されていること
   - `modelzoo-version` が、`platform-version` と異なる場合に、検証済みの Model Zoo リリースを明示的に選択すること。省略された場合、Model Zoo の解決はデフォルトで `platform-version` になります。
   - 公開されている C++ 型のレイアウトまたはエクスポートされたバイナリコントラクトが互換性のない方法で変更された場合は、`abi-version` を `deps/manifest.json` でインクリメントすること。すべての C++ アプリケーションと Python バインディングは、その ABI に対して再構築されます。
   - `CHANGELOG.md` に `## [x.y.z]` のエントリがあること
   - リリース/タグの本文にリリースノートが準備されていること

リリースフローに「既知のクラッシャー」リストは許可されません。クラッシュ回帰が発生した場合、修正されるまでリリースはブロックされます。

## 必要なステータスチェック

以下のチェックは、リリース PR およびリリースタグに対して必要です。

- `repo-hygiene`
- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `model-archive-security-gate`
- `install-smoke`
- `perf-regression-gate`
- `zero-skip-gate`
- `soak-weekly`（リリースタグに必要）
- `fuzz-nightly`（リリース候補に必要）
- `stress-gate`
- `asan-ubsan-gate`
- `release-policy-check`

これらのチェックは、以下に実装されています。

- `.github/workflows/release-gate.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/model-archive-security.yml`
- `.github/workflows/install-smoke.yml`
- `.github/workflows/perf-regression.yml`
- `.github/workflows/zero-skip.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

重複するゲート実行を避けるために、所有権をトリガーします。

- `main` への非リリース PR は、それぞれのスタンドアロンワークフローから `model-archive-security`、`install-smoke`、`perf-regression`、および `zero-skip` を実行します。
- リリース PR（`release/*` ヘッドリファレンス）およびリリースリファレンス（`release/**`、`v*`）は、`.github/workflows/release-gate.yml` から同じ処理フローを実行します。

## GitHub ブランチとタグの保護

GitHub リポジトリの設定を行います。

1. `main` を保護します。
   - マージ前にプルリクエストを必須とします。
   - 少なくとも 1 人のコードオーナーの承認を必須とします（利用可能な場合は 2 人を推奨します）。
   - 新しいコミットで保留中の承認を却下します。
   - すべての必須ステータスチェックを必須とします。
   - 強制プッシュを禁止します。
   - スカッシュのみまたは線形履歴を使用します。
2. リリースタグを作成できるユーザーを制限するために、`v*` タグを保護します。

## リリースフロー

1. 正常な `main` から `release/x.y.z` を切り出します。
2. 非リリース PR のマージを一時停止します。
3. リリースブランチでリリースゲートワークフローを実行します。
4. 候補の検証のために `vX.Y.Z-rcN` タグを作成します。
5. 最終的な `vX.Y.Z` タグに昇格します。
6. リリースブランチを `main` に高速でマージします。
7. リリースノートを公開し、リリース後のフォローアップの問題を投稿します。

## 運用上の注意点

- 汚染されたブランチからのリリースは行わないでください。
- レビューされていないコードからのリリースは行わないでください。
- 必須チェックが失敗している場合のリリースは行わないでください。
- ローカルのクラッシュ/正当性ゲートが失敗した場合、プッシュは許可されません。
- 衛生上の問題が発生した場合、手動による回避パスは許可されません。

## パフォーマンス回帰契約

- パフォーマンスゲートのエントリポイントは `scripts/ci/run_perf_regression_gate.sh` です。
- ベースラインは、`tests/perf/baselines/v2/modalix_default/` の下でプロファイルごとにスコープされます。
  - `profile.json` は、固定された Modalix 環境契約を定義します。
  - シナリオ ID ごとに 1 つのシナリオファイル（`<scenario_id>.json`）。
- 必須のシナリオ：
  - `runtime_session_sync_rgb`
  - `runtime_session_async_rgb`
  - `runtime_graph_fanout`
  - `runtime_graph_join_bundle`
  - `runtime_codec_mjpeg_decode`
  - `runtime_codec_h264_decode`
  - `runtime_codec_h265_decode`
  - `runtime_model_archive_load`
- すべてのパフォーマンス実行は、`build-perf-gate/perf_results/` に、シナリオごとの結果ファイルを公開します。
- 各結果には、以下を含める必要があります。
  - `scenario_id`
  - `modalix_profile_id`
  - `status`
  - `failure_class`
  - `reason_code`
  - `metrics`
  - `run_meta`
  - `timestamp`
- `REGRESSION`、`HARNESS_ERROR`、または `ENV_BROKEN` のいずれかの分類は、処理フローをブロックします。
