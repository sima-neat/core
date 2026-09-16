# 027 INT8テンソルでMLAのみを実行

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15分 |
| Model | yolo26n-det-int8-b1 (Model Zoo, compiled for direct MLA input and output) |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

デフォルトのPCIeルートはFP32テンソルをカードに送信し、カード上で入力が量子化され、MLAが実行され、出力が逆量子化されてFP32としてホストに返されます。すでにINT8データを保持しているアプリケーション、または量子化ステップを自分で制御したいアプリケーションは、`ModelOptions.mla_only`を設定できます。カードはMLAのみを実行します。ホストはMLAの入口コントラクトに一致するINT8テンソルを送信し、生のINT8ヘッドを受け取ります。`model.info()`はすべてのテンソルの量子化パラメータを公開するため、ホストは1つの式で量子化と逆量子化を行えます。

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

このチュートリアルではINT8の場合を示します。`mla_only`は互換性のあるBF16アーカイブにも対応しており、BF16テンソルには量子化パラメータがありません。

## Walkthrough

1つのプログラムがホスト上で画像を量子化し、MLAのみのルートを実行し、ヘッドを逆量子化して、同じキュー上のデフォルトルートと照合します。

### MLAのみのコントラクトを確認する {#step-inspect-contract}

`mla_only`を有効にして`Model`を構築します。`info()`はINT8の入力と出力を報告し、それぞれに`quant.scale`と`quant.zero_point`が含まれます。リファレンスモデルの入力は`images`の1つで、INT8の`[640, 640, 3]` HWCテンソルです。

### ホスト上で量子化する {#step-quantize-on-host}

リファレンスモデルは、ピクセルが`[0, 1]`の範囲にある1枚のRGB画像を期待します。画像を`640x640`にリサイズし、BGRをRGBに変換し、255で割り、入口のパラメータで量子化の式を適用します。この前処理はアーカイブのコントラクトではなくモデルに属するものです。別のモデルには独自のレシピが必要です。同じコードを逆量子化した値を保持しておきます。これは、デフォルトルートが同条件の比較に必要とする正確なFP32入力です。

### INT8ルートを実行する {#step-run-int8}

`build()`はMLAのみを含むカードのパイプラインを起動します。`run()`はINT8テンソルを受け付け、`info().outputs`が報告した順序と名前で、出力ごとに1つの密なINT8テンソルを返します。このルートは他のdtypeを拒否します。FP32のプッシュはカード上で量子化されるのではなく、失敗します。

### 逆量子化して比較する {#step-dequantize-and-compare}

`mla_only`なしで2つ目の`Model`を構築し、逆量子化したFP32の値をデフォルトルートに送信します。各出力のパラメータでINT8ヘッドを逆量子化し、ヘッドごとの最大偏差をそのヘッドのスケール単位で出力します。両方のルートは同じコードで同じMLAプログラムを実行するため、誤差はゼロです。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているように、チュートリアルバンドルをダウンロードします。

このチュートリアルは、MLAの入出力を直接扱うようにコンパイルされたModel ZooのYOLO26n INT8アーカイブを実行します。抽出したPCIeエクストラのルートにダウンロードします。

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
```

その他のアーカイブは、Model SDKの`tessellate_parameters`で`enable_mla=True`を指定し、すべての入力に`HWC`のDRAMレイアウト、すべての出力に`HWC16`を指定してコンパイルされていれば条件を満たします。Model Zooの`yolo_v8s`ビルドのように代わりにCVUでテッセレーションを行うアーカイブは、`mla_only`を有効にすると拒否されます。

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model yolo26n-det-int8-b1.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model yolo26n-det-int8-b1.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model yolo26n-det-int8-b1.tar.gz
```

両方のバージョンがコントラクトと、すべてのヘッドでゼロの偏差を出力します。

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00390434 zero_point=-128
  output bbox_0 INT8 [80, 80, 4] scale=0.0302856 zero_point=-117
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 4] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

デフォルトはカード0とキュー0です。別のカードを使用する場合にのみ`--card N`を渡します。

## In Practice

INT8アーカイブでは、アプリケーションが量子化を所有する場合に`mla_only`を有効にします。センサーや前段のモデルからすでにINT8を生成している場合、独自の後処理のために生のINT8ヘッドが必要な場合、またはカード側のレイテンシから量子化と逆量子化のステージを取り除きたい場合です。すべてのスケールとゼロポイントは`model.info()`から読み取り、モデルの別のビルドからコピーしないでください。

このルートは全か無かです。すべての入力は対応する`info().inputs`エントリのdtype、形状、バイトサイズに一致する必要があり、画像の前処理やボックスデコードを`mla_only`と組み合わせることはできません。ホストがFP32データを保持し、変換を自分で行う必要がない場合は、デフォルトルートを使用してください。

デプロイメントの診断については、[PCIeモデルワークフロー](/develop-apps/development-workflow/pcie-model/)に進んでください。

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
