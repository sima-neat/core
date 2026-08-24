---
title: "sccache 活用ガイド"
description: "Neat コンパイラキャッシュをローカル環境と Vulcan 環境で利用し、問題が発生した場合はトラブルシューティングを行います。"
sidebar_position: 2
slug: /develop-apps/contribute/sccache
---

# Neat `sccache` チートシート

Neat は、C および C++ コンパイラの起動に [`sccache`](https://github.com/mozilla/sccache) を使用します。最終的な Neat パッケージ、テスト結果、依存関係のダウンロード、または Docker イメージではなく、コンパイラの出力をキャッシュします。

ほとんどの開発者にとって、インストールや設定は必要ありません。通常どおり `build.sh` を使用し、最後に表示されるキャッシュ統計を確認してください。

## 概要

| ビルド | キャッシュレベル | 書き込み先 | `--clean` 後も残るか |
|---|---|---|---|
| ローカル | ユーザーローカルディスク | ローカルディスク | はい |
| Vulcan `develop` または `main` | ランナーローカルディスク、次に保護されたブランチの S3 | ローカルディスクとその保護された S3 名前空間 | S3 には残る |
| Vulcan のフィーチャーブランチプッシュ | ランナーローカルディスク、次にブランチの S3 | ローカルディスクとその分離されたブランチの名前空間 | ブランチの削除まで |
| Vulcan のタグまたは直接参照ではない参照 | ランナーローカルディスク、次に最も近い保護された S3 | ローカルディスクのみ | 永続的なランナーの状態は保存されない |

サポートされているエントリポイントは常に次のとおりです。

```bash
./build.sh <options>
```

コンパイラを置き換えるために、または起動オプションを手動で追加するために、`sccache` を使用しないでください。`build.sh` は、CMake の両方の起動オプションを提供します。

```text
CMAKE_C_COMPILER_LAUNCHER
CMAKE_CXX_COMPILER_LAUNCHER
```

## ローカルビルド

### 通常の使用

`auto` モードでキャッシュが有効になっています。

```bash
./build.sh --dev-only
./build.sh --all --clean
```

デフォルトのキャッシュの保存場所と上限は次のとおりです。

```text
~/.cache/sima-neat/sccache
10 GiB
```

キャッシュは`build/`の外部にあります。`build/`を削除したり、`--clean`を実行したりしても、キャッシュされたコンパイラ出力は削除されません。

`sccache`が`PATH`に含まれていない場合、`build.sh`は、指定されたバージョンのリリースを次の場所にダウンロードします。

```text
${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/tools/sccache/<version>/
```

アーカイブは、`scripts/configure_sccache.sh` に記録されている SHA-256 を使用して検証されます。arm64 および x86-64 アーキテクチャの Linux および macOS がサポートされます。

### 共通の制御項目

```bash
# Explicitly require sccache. Fail the build if it cannot be configured.
SIMANEAT_SCCACHE=on ./build.sh --all

# Disable caching for a reproducibility comparison.
SIMANEAT_SCCACHE=off ./build.sh --all --clean

# Put the local cache on a larger or faster volume.
SCCACHE_DIR=/mnt/nvme/sccache ./build.sh --all

# Change the local cache limit.
SCCACHE_CACHE_SIZE=20G ./build.sh --all
```

`SIMANEAT_SCCACHE=auto` がデフォルト設定です。このモードでは、ブートストラップの失敗時に警告が表示され、キャッシュを使用せずにビルドが続行されます。`on` を設定すると、その失敗が致命的なものとなります。

### ローカルキャッシュの確認またはクリア

`build.sh` で選択されたものと同じバイナリを使用するか、`sccache` が `PATH` に設定されている場合は、それを使用します。

```bash
sccache --show-stats
sccache --zero-stats
sccache --show-adv-stats
```

スペースを確保するには、サーバーを停止し、設定済みのキャッシュディレクトリのみを削除してください。

```bash
sccache --stop-server
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/sccache"
```

削除する前に、解決されたパスを確認してください。ユーザーのキャッシュディレクトリ全体を削除しないでください。

## Vulcan Cloud Builds

Vulcanは、ローカルディスクキャッシュに加えて、暗号化されたS3レベルのキャッシュを提供します。

```text
s3://sima-neat-compiler-cache-production/
  core/
    sccache-v1/
      <architecture>/
        <sdk-cache>/
          <build-mode>/
            develop/
              branches/<encoded-feature-branch>/
            main/
              branches/<encoded-feature-branch>/
```

たとえば：

```text
core/sccache-v1/arm64/sdk-develop/standard/develop/
```

名前空間には、意図的に以下のものが含まれます。

- `sccache-v1`: キャッシュスキーマ。意図的なグローバルリセットを可能にします。
- architecture: arm64とx86-64のコンパイラ出力が混ざるのを防ぎます。
- SDKキャッシュID: 互換性のないSDK/ツールチェーン出力が混ざるのを防ぎます。
- ビルドモード: 標準とファジングのインストルメンテーションを分離します。
- 保護されたベースブランチ: `develop`と`main`が同じ名前空間に書き込むのを防ぎます。

S3バケットはプライベートで、独自のKMSキーで暗号化され、アーティファクトバケットとは分離されています。コンパイラキャッシュオブジェクトはプライベートで使い捨てであるため、CloudFrontディストリビューションはありません。オブジェクトは45日後に自動的に期限切れになります。

### ブランチへのアクセス

| Gitリファレンス | OIDCロール | S3モード |
|---|---|---|
| 正確な`refs/heads/develop` | 保護された書き込み権限 | `develop/`内の`READ_WRITE` |
| 正確な`refs/heads/main` | 保護された書き込み権限 | `main/`内の`READ_WRITE` |
| 直接フィーチャーブランチへのプッシュ | ブランチ書き込み権限 | `<base>/branches/<branch>/`以下の`READ_WRITE` |
| タグまたは間接的なリファレンス | 読み取り専用 | 選択された保護されたベースラインからの`READ_ONLY` |

フィーチャーブランチは、最初のビルド時に、最も近い保護されたGitの祖先（`develop`または`main`）からキャッシュを自身の名前空間にコピーします。その後、ビルドは、そのブランチの名前空間のみを読み書きします。その後のビルドでは、GitHubのブランチ削除イベントによって、そのブランチのすべてのアーキテクチャ、SDK、およびビルドモードの名前空間が削除されるまで、それを再利用します。フィーチャーブランチは、いずれかの保護されたキャッシュに書き込むことはできません。

自動的な祖先検出は、`develop`と`main`へのマージベース距離を比較します。再利用可能な、または手動で実行できるワークフローでは、例外的なブランチが明示的なベースラインを必要とする場合に、`cache_base_branch=develop|main`を設定できます。AWS認証情報は、短期間有効なGitHub OIDC認証情報であり、長期間有効なAWSキーはGitHubまたはSDKコンテナに保存されません。

最初の書き込み可能な保護されたブランチのビルドの前に、空の保護された名前空間が予想されます。フィーチャーブランチは、自身の名前空間にデータを入力できますが、選択された保護されたベースラインが空の場合、初期のキャッシュヒットは発生しません。

Vulcanは、CMakeを構成する前に、明示的に`sccache`の起動をチェックします。S3、KMS、ネットワーク、または一時的な認証情報がキャッシュサーバーの起動を妨げる場合、ワークフローは警告を出力し、`sccache`なしでコンパイルします。したがって、リモートキャッシュの可用性は最適化であり、コンパイルをブロックすることはできません。Vulcanランナーは一時的なものであるため、ランナーローカルキャッシュはフォールバックとして使用されません。ローカル開発ビルドは、通常の永続ディスクキャッシュを保持します。

## ビルド統計の読み取り

キャッシュされたすべてのビルドは、次のような出力で終了します。

```text
Compile requests                    623
Cache hits                          619
Cache misses                          4
Cache hits rate                   99.36 %
Cache timeouts                        0
Cache read errors                     0
Cache write errors                    0
Compilations                          4
```

重要なフィールドの意味は以下のとおりです。

| フィールド | 意味 |
|---|---|
| コンパイル要求 | `sccache` が処理したコンパイラの呼び出し |
| キャッシュヒット | コンパイラを実行せずにキャッシュから復元された要求 |
| キャッシュミス | コンパイルが必要だった要求 |
| コンパイル | 実際に実行されたコンパイラプロセス |
| キャッシュできない呼び出し | `sccache` が意図的にバイパスした呼び出し |
| 読み取り/書き込みエラー | キャッシュバックエンドの障害。ローカルまたは書き込み可能なビルドでゼロ以外の値が表示された場合は調査してください |
| キャッシュの場所 | アクティブなバックエンド（ローカルディスクや多層キャッシュなど） |

新しいツールチェーン、SDKキャッシュ、アーキテクチャ、ビルドモード、または大幅に変更されたソースツリーの最初のビルドでは、ヒット率が低いのは正常です。同じコミットと設定で2回目のビルドを行い、キャッシュのパフォーマンスを評価してください。

いずれかのレベルが読み取り専用の場合、`sccache` v0.16 は、読み取り専用ビルドが成功した場合でも、書き込みの試行を書き込みエラーとして報告する場合があります。これは、タグやその他の間接的なコンテキストに適用されます。直接的なフィーチャーブランチのプッシュでは、`READ_WRITE` が報告されるはずです。これらのビルドで書き込みエラーが発生した場合は、調査してください。

## 簡単な検証

### ローカルでの再利用の検証

同じクリーンビルドを2回実行します。

```bash
SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist

SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist
```

2回目の実行では、成功率が大幅に向上するはずです。正確な数値は、ソースの変更、コンパイラのプローブ、生成されたファイル、および選択されたビルドによって異なります。

### Vulcan の設定を確認する

GitHub Actions のビルドログで、次の項目を探してください。

```text
sccache enabled: sccache <version>
sccache local cache: <path> (<limit>)
sccache remote cache: s3://<bucket>/<prefix> (READ_ONLY|READ_WRITE)
```

次に、最終的な統計に予期しない読み取り、書き込み、タイムアウト、またはキャッシュエラーが含まれていないことを確認します。

## トラブルシューティング

### `sccache` が有効になっていない

- ビルドで `build.sh` が使用されていることを確認します。
- `SIMANEAT_SCCACHE` が `off` に設定されていないことを確認します。
- `SIMANEAT_SCCACHE=on` を設定して再実行し、ブートストラップエラーが致命的なエラーになるようにします。
- `curl`、`tar`、および `sha256sum` または `shasum` が利用可能であることを確認します。

### 2回目のローカルビルドでもまだキャッシュにヒットしない

- 2つのビルドで同じコンパイラ、SDK、ビルドモード、およびフラグが使用されていることを確認します。
- `SCCACHE_DIR` が同じ永続的なディレクトリを指していることを確認します。
- タイムスタンプまたは変更される絶対パスを含む生成された入力がないか確認します。
- `Non-cacheable calls` と `Unsupported compiler calls` を確認します。
- `SCCACHE_CACHE_SIZE` によってキャッシュが削除されていないことを確認します。

### Vulcan でリモートヒットがゼロと表示される

- 選択した `develop` または `main` ベースラインが入力されていることを確認します。
- 機能ブランチの場合、ログに予想されるベースブランチとそのエンコードされたブランチ固有のプレフィックスが報告されていることを確認します。
- アーキテクチャ、SDK キャッシュ ID、およびビルドモードを比較します。
- ログに予想されるバケットとプレフィックスが表示されていることを確認します。
- コールドネームスペースを通常どおり扱い、2つの同一のビルドを比較します。

### S3 の起動時に `AccessDenied` が発生する

キャッシュロールには、`.sccache_check` プローブ用にプレフィックス範囲の `s3:ListBucket` が必要であり、さらにオブジェクトの権限が必要です。

- リーダー: `GetObject`
- ライター: `GetObject` および `PutObject`

両方のロールには、対応する KMS 権限も必要です。回避策として、EC2 ランナーロールに S3 または KMS 権限を追加しないでください。代わりに、Vulcan の GitHub OIDC キャッシュロールを修正してください。

### コンパイラエラーの診断中にキャッシュをバイパスする

```bash
SIMANEAT_SCCACHE=off ./build.sh --all --clean
```

エラーが解消されない場合、それはキャッシュされたコンパイラの出力が原因ではありません。

## 所有権と信頼できる情報源

| 懸念事項 | 情報源 |
|---|---|
| ローカルブートストラップ、バージョン、チェックサム、キャッシュのデフォルト設定 | `scripts/configure_sccache.sh` |
| CMakeランチャーの統合と統計 | `build.sh` |
| ブランチの役割選択とキャッシュのネームスペース | `.github/workflows/vulcan-ci.yml` |
| 再利用可能なVulcanワークフローの入力とOIDCの設定 | `sima-neat/.github` |
| S3、KMS、ライフサイクル、およびキャッシュのIAMロール | `sima-neat/vulcan` |

キャッシュスキーマ、ツールチェーンの互換性、またはアクセスモデルを変更する場合は、影響を受けるすべてのリポジトリとこのページをまとめて更新してください。
