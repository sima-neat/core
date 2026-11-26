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
7. Vulcan CI（`.github/workflows/vulcan-ci.yml`）が、タグを付けるコミットそのもので合格していること。
8. 厳格なテストレーンに対して、ゼロスキップゲートが緑色であること（`zero-skip-gate`）。
9. 必要なガバナンスファイルが存在し、有効であること：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
10. リリースメタデータが完了していること：
   - タグを付けるコミットの `deps/manifest.json` の `package-version` がリリースバージョンと一致していること（たとえば `v0.7.0` なら `0.7.0`）
   - 必要に応じて、`platform-version` が `deps/manifest.json` に更新されていること
   - `modelzoo-version` が、`platform-version` と異なる場合に、検証済みの Model Zoo リリースを明示的に選択すること。省略された場合、Model Zoo の解決はデフォルトで `platform-version` になります。
   - `abi-version` は、公開 C++ 型のレイアウトまたはエクスポートされたバイナリコントラクトに互換性のない変更があり、
     かつ `develop` の ABI バージョンが `main` と同じ場合にのみ、`deps/manifest.json` でインクリメントすること。
     `develop` が次のリリースに向けてすでに高い ABI バージョンを持つ場合、リリース前の追加の ABI 非互換変更でもそのバージョンを維持すること。
     互換性のない変更後は、一致するヘッダーとライブラリを使って、すべての C++ アプリケーションと Python バインディングを再ビルドすること。
   - リリース/タグの本文にリリースノートが準備されていること

リリースフローに「既知のクラッシャー」リストは許可されません。クラッシュ回帰が発生した場合、修正されるまでリリースはブロックされます。

## 必要なステータスチェック

リリースタグを作成する前に、タグを付けるコミットそのもので Vulcan CI（`.github/workflows/vulcan-ci.yml`）が
合格している必要があります。確認方法はリリースフローの手順 5 に記載しています。

以下のチェックも、リリース PR およびリリースタグに対して必要です。

- `configure-build-sanity`
- `docs-link-check`
- `zero-skip-gate`

## スケジュール実行のテストレーン

これらのレーンはリリースをブロックしません。クラッシュ/正当性、ストレステスト、サニタイザー、ファズテストは毎晩実行されます。
ソーク安定性テストと長時間テストは毎週実行されます。

- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`

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

`v0.7.0` のような正式なタグでリリースします。リリース候補のタグは作成しません。Release ワークフロー
（**Actions → Release**）は `release-X.Y` を作成または再利用し、その最新のコミットにタグを付けて、GitHub リリースの下書きを作成します。
以下の手順は、新しいラインで 0.7.0 をリリースする場合です。既存のラインでのパッチリリース（たとえば 0.6.3）では、手順 3 と 6 が異なります。

1. `develop` で、`deps/manifest.json` の `package-version` を `0.7.0` に設定します。リリースをブロックする条件で必要な場合は `platform-version`、`modelzoo-version`、`abi-version` を更新し、「最新ドキュメントのバージョン」の説明に従ってリリースデータを更新します。
2. プルリクエストで `develop` を `main` にマージします。
3. タグを付けるコミットを特定します。先に `git fetch origin` を実行します。
   - 新しいライン：ワークフローは `main` の最新のコミットから `release-0.7` を作成します。コミットは `git rev-parse origin/main` です。
   - 既存のライン：プルリクエストで `main` を `release-0.6` にマージします。`branch_exists_behavior` の既定値が `reuse` であるため、ワークフローは `release-0.6` を再利用し、その最新のコミットにタグを付けます。コミットは `git rev-parse origin/release-0.6` です。

   タグが作成されるまで、そのブランチには何もマージしません。
4. そのコミットのバージョンを確認します。`git show <commit>:deps/manifest.json` に `"package-version": "0.7.0"` が表示される必要があります。
5. そのコミットそのもので Vulcan CI が合格していることを確認します。`gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>` で実行の一覧を確認できます。`main` またはリリースブランチ上の成功した実行も有効です。なければ、手順 3 のブランチで `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main` を実行して Vulcan CI を開始します。パッチリリースでは `--ref release-0.6` を使います。新しい実行の `headSha` がそのコミットであることを確認し、`gh run watch <run-id> --repo sima-neat/core --exit-status` で完了を待ちます。そのコミットでの実行が合格するまでタグを付けません。
6. **Actions → Release → Run workflow** を開きます。`release_line` を `0.7`、`tag` を `0.7.0` に設定し、`main` を `source_branch` のままにして、`prerelease` はオフ、`latest` はオンのままにします。0.6.3 の場合は `release_line` を `0.6`、`tag` を `0.6.3` に設定し、新しいラインがすでにリリースされていれば `latest` をオフにします。ワークフローは `v0.7.0` タグを付け、リリースの下書きを作成します。Target Release が `core-0.7.0` のすべての issue が Done であることが必要で、その後それらの Release Status を `released` に設定します。実行サマリーの **Target SHA** が手順 3 のコミットであることを確認します。一致しない場合は、リリースの下書きを公開しません。
7. リリースの下書きにリリースノートを記載して公開し、リリース後のフォローアップの issue を作成します。

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
