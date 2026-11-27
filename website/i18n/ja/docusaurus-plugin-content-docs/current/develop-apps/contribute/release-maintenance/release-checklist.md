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
8. Vulcan CI のテストレーンで、ゼロスキップゲート（`scripts/ci/run_zero_skip_gate.sh`）に合格していること。
9. 必要なガバナンスファイルが存在し、有効であること：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-maintenance/release-checklist.md`
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

`v0.7.0` のような正式なタグでリリースします。リリース候補のタグは作成しません。メジャーリリースとパッチリリースは
同じワークフロー **Actions → Release** を使います。このワークフローは、新しいラインでは `main` から `release-X.Y` を作成し、
既存のラインでは既存の `release-X.Y` を再利用して、そのブランチの最新のコミットにタグを付け、GitHub リリースの下書きを作成します。
最初のジョブでは、そのコミットの `package-version` が `tag` と等しいことを確認します。Target Release が `core-X.Y.Z` の
すべての issue が Done であることが必要で、その後それらの Release Status を `released` に設定します。

### メジャーリリース

この例では、新しいラインで `v0.7.0` をリリースします。

1. `develop` で、`deps/manifest.json` の `package-version` を `0.7.0` に設定します。リリースをブロックする条件で必要な場合は `platform-version`、`modelzoo-version`、`abi-version` を更新し、「最新ドキュメントのバージョン」の説明に従ってリリースデータを更新します。
2. プルリクエストで `develop` を `main` にマージします。
3. `git fetch origin` を実行します。タグを付けるコミットは `git rev-parse origin/main` です。タグが作成されるまで、`main` には何もマージしません。
4. そのコミットで「タグ付けの前」のチェックを実行します。
5. **Actions → Release → Run workflow** を開きます。`release_line` を `0.7`、`tag` を `0.7.0` に設定し、`main` を `source_branch` のままにして、`prerelease` はオフ、`latest` はオンのままにします。ワークフローは `main` から `release-0.7` を作成し、`v0.7.0` タグを付けて、リリースの下書きを作成します。
6. 「タグ付けの後」と「公開」に進みます。

### パッチリリース

この例では、既存の `release-0.6` で `v0.6.3` をリリースします。

1. `release-0.6` を準備します。
   - `develop` を `main` 経由でマージしたときに `release-0.6` に入るすべての変更が `0.6.3` 向けである場合は、そのマージの手順を使います。メジャーリリースの手順 1 と同様に `develop` で `package-version` を `0.6.3` に設定し、`develop` を `main` にマージしてから、プルリクエストで `main` を `release-0.6` にマージします。
   - それ以外の場合は、`release-0.6` からブランチを作成し、意図した修正だけを `git cherry-pick -x` で取り込み、`package-version` を `0.6.3` に設定して、`release-0.6` へのプルリクエストを作成します。
2. `git fetch origin` を実行します。タグを付けるコミットは `git rev-parse origin/release-0.6` です。タグが作成されるまで、`release-0.6` には何もマージしません。
3. そのコミットで「タグ付けの前」のチェックを実行します。
4. **Actions → Release → Run workflow** を開きます。`release_line` を `0.6`、`tag` を `0.6.3` に設定し、`main` を `source_branch` のままにして、`prerelease` はオフのままにします。新しいラインがすでに公開されている場合は `latest` をオフにします。`branch_exists_behavior` の既定値が `reuse` であるため、ワークフローは `release-0.6` を再利用し、その最新のコミットに `v0.6.3` タグを付けて、リリースの下書きを作成します。
5. 「タグ付けの後」と「公開」に進みます。

### 共通の手順

`<commit>` は、メジャーリリースの手順 3 またはパッチリリースの手順 2 で特定した、タグを付けるコミットです。
コマンドでは `0.7.0` を使っています。パッチリリースでは、そのバージョン（たとえば `0.6.3`）を使います。

#### タグ付けの前

1. バージョンを確認します。`git show <commit>:deps/manifest.json` に `"package-version": "0.7.0"` が表示される必要があります。表示されない場合、Release ワークフローは最初のジョブで、ブランチやタグを作成する前に停止します。
2. そのコミットそのもので標準の Vulcan CI が合格していることを確認します。`gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>` で実行の一覧を確認でき、`main` またはリリースブランチ上の既存の成功した実行を再利用できます。なければ、`gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main` で実行を開始します。パッチリリースでは `--ref release-0.6` を使います。新しい実行の `headSha` がそのコミットであることを確認し、`gh run watch <run-id> --repo sima-neat/core --exit-status` で完了を待ちます。
3. issue ゲートを確認します。Neat Releases プロジェクト（`https://github.com/orgs/sima-neat/projects/6`）で、Target Release が `core-0.7.0` の issue が少なくとも 1 つあり、その Target Release を持つすべての issue が Done である必要があります。`dry_run` をオンにした Release の実行では、何もプッシュせず issue も変更せずに、このチェックとバージョンのチェックを行います。

#### タグ付けの後

1. Release の実行サマリーの **Target SHA** がそのコミットであることを確認します。一致しない場合は、リリースの下書きを公開しません。
2. `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref v0.7.0` でタグに対して標準の Vulcan CI を開始します。ワークフローが付けたタグだけでは、ワークフローの実行は開始されません。この実行により、`sima-cli neat install core@v0.7.0` がインストールするバージョン付きのパッケージとオフラインバンドルが公開されます。
3. `gh run watch <run-id> --repo sima-neat/core --exit-status` で、`Publish develop and release offline bundles` ジョブを含む実行全体が成功するまで待ちます。`latest.tag` はこのジョブの開始前に更新されるため、実行の完了を示すものではありません。

#### 公開

1. GitHub リリースの下書きを開き、リリースノートを確認して **Publish release** をクリックします。
2. リリース後のフォローアップの issue を作成します。

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
