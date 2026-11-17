---
title: "Model Browser"
sidebar_position: 4
---

# Model Browser

[Palette Model Browser](https://developer.sima.ai/models/) には、SiMa.ai が Modalix 向けにコンパイルおよび測定したモデルが一覧表示されます。各モデルのページには、Modalix DevKit で測定したパフォーマンスと精度が表示され、コンパイル済みモデルをダウンロードできます。

次のような場合に利用してください。

- 物体検出やテキスト生成など、タスクに合ったモデルを探す
- モデルとそのバリアントのパフォーマンスと精度を比較する
- Neat ですぐに実行できるコンパイル済みモデルをダウンロードする
- 公開されているスクリプトからモデルを自分で再ビルドする

## sima-cli でモデルをダウンロードする

`sima-cli` を使うと、同じモデルをターミナルから参照できます。アカウントは不要です。

モデルを検索します。

```bash
sima-cli models list -q rf_detr_nano
```

モデルを選択し、次にバリアントを選択します。バリアントはモデルのコンパイル方法の 1 つで、重みとアクティベーションのビット幅にちなんだ名前が付いています。コマンドは、測定された精度とパフォーマンスを含むモデルカードを表示し、モデルをダウンロードするかどうかを尋ねます。

ID とバリアントがわかっている場合は、モデルを直接ダウンロードします。

```bash
sima-cli models download --id rf_detr_nano --variant w8a8_w16a16
```

コマンドはモデルを `models/<model id>/<variant>/` に保存します。別の場所に保存するには、`--output <folder>` を追加します。

ライセンスで再配布が禁止されているモデルはダウンロードできません。次のセクションで説明するとおり、スクリプトから再ビルドしてください。

コマンドの詳細については、[`sima-cli models`](/tools/sima-cli/models/) リファレンスを参照してください。

## モデルを再現する

すべてのモデルには、そのモデルをビルドおよび測定したスクリプトが付属しており、[models リポジトリ](https://github.com/sima-neat/models)で公開されています。各モデルには `models/<model id>/` フォルダーがあり、ステップごとに 1 つのスクリプトがあります。

| ステップ | スクリプト | 実行場所 |
| --- | --- | --- |
| 公式モデルを ONNX にエクスポートする | `convert.py` | お使いのコンピューター |
| 量子化とコンパイルを行う | `compile.py` | Palette SDK の Model Compiler |
| パフォーマンスを測定する | `performance.py` | Modalix DevKit |
| 精度を測定する | `accuracy.py` | Modalix DevKit |

同じフォルダーにある `model.yaml` ファイルには、ビルドできるバリアントが `build_profiles` の下に、また各ステップに必要なデータセットが記載されています。

[入門ガイド](https://developer.sima.ai/models/docs/getting-started)では、コピーして使えるコマンドとともにすべてのステップを順に説明し、結果を公開値と比較する方法を示します。
