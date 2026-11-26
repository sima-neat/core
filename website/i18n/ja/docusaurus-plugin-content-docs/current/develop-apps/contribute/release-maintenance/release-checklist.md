---
title: "リリースチェックリスト"
description: "リリースゲートポリシーと再現可能なリリース手順"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# リリースチェックリスト

このドキュメントは、正式なリリースゲートポリシーです。

## 最新ドキュメントのバージョン

`deps/manifest.json` を更新した後、リリースに合わせて `website/release-data/current.json` を一度更新します。`npm --prefix website run sync:release-readmes` を実行し、具体的なバージョンを含む生成済み README コマンドをコミットします。`npm --prefix website run check:releases` と `./build.sh --doc` を実行して、トークン、過去のバージョンの例外、翻訳、レンダリングされたコマンドを検証します。互換性ページは具体的なバージョンを使って手動で管理します。過去のリリースノートのバージョンは固定したままにします。リリースデータだけの変更では、翻訳やソースハッシュの更新は不要です。各フィールドの管理責任とトークンの規則は `website/release-data/README.md` に記載されています。

## リリースをブロックする条件

以下のすべての条件が満たされている場合にのみ、リリースが許可されます。

1. トラッキングされているソース/ドキュメントファイルの変更競合マーカーがないこと。
2. 公開ドキュメントに対する命名規則のチェックに合格すること。
3. 構成/ビルドの整合性チェックに合格すること（`cmake -S . -B ...` および `cmake --build ... --target sima_neat`）。
4. ドキュメントのリンクチェックが厳格なモードで合格すること（`DOCS_STRICT_LINKS=1`）。
5. 生成ステップの後に、作業ツリーがクリーンな状態であること。
6. リリース前に、未解決のクラッシュ/正当性に関する問題がゼロであること、およびリリースリファレンス上でもゼロであること。
7. クラッシュ/正当性/ストレステスト/サニタイザーゲートが、リリースリファレンス上ですべて緑色であること。
8. リリースタグに対するソーク安定性テストが緑色であること（`soak-weekly`）。
9. リリース候補に対するファズテストが緑色であること（`fuzz-nightly`）。
10. 厳格なテストレーンに対して、ゼロスキップゲートが緑色であること（`zero-skip-gate`）。
11. 必要なガバナンスファイルが存在し、有効であること：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-maintenance/release-checklist.md`
12. リリースメタデータが完了していること：
   - 必要に応じて、`package-version` と `platform-version` が `deps/manifest.json` に更新されていること
   - `modelzoo-version` が、`platform-version` と異なる場合に、検証済みの Model Zoo リリースを明示的に選択すること。省略された場合、Model Zoo の解決はデフォルトで `platform-version` になります。
   - `abi-version` は、公開 C++ 型のレイアウトまたはエクスポートされたバイナリコントラクトに互換性のない変更があり、
     かつ `develop` の ABI バージョンが `main` と同じ場合にのみ、`deps/manifest.json` でインクリメントすること。
     `develop` が次のリリースに向けてすでに高い ABI バージョンを持つ場合、リリース前の追加の ABI 非互換変更でもそのバージョンを維持すること。
     互換性のない変更後は、一致するヘッダーとライブラリを使って、すべての C++ アプリケーションと Python バインディングを再ビルドすること。
   - リリース/タグの本文にリリースノートが準備されていること

リリースフローに「既知のクラッシャー」リストは許可されません。クラッシュ回帰が発生した場合、修正されるまでリリースはブロックされます。

## 必要なステータスチェック

以下のチェックは、リリース PR およびリリースタグに対して必要です。

- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `zero-skip-gate`
- `soak-weekly`（リリースタグに必要）
- `fuzz-nightly`（リリース候補に必要）
- `stress-gate`
- `asan-ubsan-gate`

これらのチェックは、以下に実装されています。

- `.github/workflows/vulcan-ci.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

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

1. **Actions → Release → Run workflow** を開いてリリース候補を作成します。たとえば 0.7.0 をリリースする場合は、`release_line` を `0.7`、`tag` を `0.7.0-rc1` に設定し、`main` を `source_branch` のままにして、`prerelease` をオン、`latest` をオフにします。このワークフローは `release-0.7` を作成または再利用し、`v0.7.0-rc1` タグを付けてプレリリースの下書きを作成します。`release-0.7` がすでに存在する場合（たとえば 0.7.1 の場合）、`branch_exists_behavior` の既定値が `reuse` であるため、ワークフローはその最新のコミットにタグを付けます。最初の候補を作成する前に、プルリクエストで `main` を `release-0.7` にマージします。
2. 非リリース PR のマージを一時停止します。
3. 候補を検証します。修正が必要な場合は、修正を `release-0.7` にマージしてから、`0.7.0-rc2` などの次の候補タグでワークフローを再実行します。
4. 候補が検証に合格したら、`release-0.7` がまだその候補を指していることを確認します。ワークフローは `release-0.7` の最新のコミットにタグを付け、それより古いコミットにはタグを付けられません。`git fetch --tags origin` の後、`git rev-parse origin/release-0.7` が `git rev-parse v0.7.0-rcN^{commit}` と同じコミットを出力する必要があります。ここで `rcN` は検証に合格した候補です。候補の後に変更が入っていたら、先に新しい候補を作成します。そのうえで `tag` を `0.7.0` に設定し、`prerelease` をオフ、`latest` をオンにしてワークフローを実行します。ワークフローはそのコミットに `v0.7.0` タグを付け、正式リリースの下書きを作成します。この実行では、Target Release が `core-0.7.0` のすべての issue が Done であることも必要で、その後それらの Release Status を `released` に設定します。候補の実行ではこのチェックを行いません。
5. リリースブランチを `main` に高速でマージします。
6. リリースノートを公開し、リリース後のフォローアップの問題を投稿します。

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
