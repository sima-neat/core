# 027 INT8テンソルでMLAのみを実行

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15分 |
| Model | any archive compiled for direct MLA input and output |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

デフォルトのPCIeルートはFP32テンソルをカードに送信し、EV74がそれを量子化し、MLAが実行され、EV74が結果をFP32に逆量子化して返します。すでにINT8データを保持しているアプリケーション、または量子化ステップを自分で制御したいアプリケーションは、`ModelOptions.mla_only`を設定できます。カードはMLAのみを実行します。ホストはMLAの入口コントラクトに一致するINT8テンソルを送信し、生のINT8ヘッドを受け取ります。`model.info()`はすべてのテンソルの量子化パラメータを公開するため、ホストは1つの式で量子化と逆量子化を行えます。

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

## Walkthrough

1つのプログラムがホスト上で画像を量子化し、MLAのみのルートを実行し、ヘッドを逆量子化して、同じキュー上のデフォルトルートと照合します。

### MLAのみのコントラクトを確認する {#step-inspect-contract}

`mla_only`を有効にして`Model`を構築します。`info()`はINT8の入力と出力を報告し、それぞれに`quant.scales[0]`と`quant.zero_points[0]`が含まれます。入力には、モデルがキャリブレーションされた浮動小数点の範囲である`input_range`も含まれます。複数の入力を持つモデルは、送信順に入力ごとに1つのINT8テンソルを列挙します。

### ホスト上で量子化する {#step-quantize-on-host}

画像を入口のジオメトリにリサイズし、BGRをRGBに変換し、ピクセルを`input_range`にマッピングして、入口のパラメータで量子化の式を適用します。同じコードを逆量子化した値を保持しておきます。これは、デフォルトルートが同条件の比較に必要とする正確なFP32入力です。

### INT8ルートを実行する {#step-run-int8}

`build()`はMLAのみを含むカードのパイプラインを起動します。`run()`はINT8テンソルを受け付け、`info().outputs`が報告した順序と名前で、出力ごとに1つの密なINT8テンソルを返します。このルートは他のdtypeを拒否します。FP32のプッシュはカード上で量子化されるのではなく、失敗します。

### 逆量子化して比較する {#step-dequantize-and-compare}

`mla_only`なしで2つ目の`Model`を構築し、逆量子化したFP32の値をデフォルトルートに送信します。各出力のパラメータでINT8ヘッドを逆量子化し、ヘッドごとの最大偏差をそのヘッドのスケール単位で出力します。両方のルートは同じコードで同じMLAプログラムを実行するため、誤差はゼロです。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているように、チュートリアルバンドルをダウンロードします。

このチュートリアルには、MLAの入力と出力を直接扱うようにコンパイルされたアーカイブが必要です。Model SDKの`tessellate_parameters`で`enable_mla=True`を指定し、すべての入力に`HWC`のDRAMレイアウト、すべての出力に`HWC16`を指定します。Model Zooのアーカイブは代わりにEV74でテッセレーションを行うため、`mla_only`を有効にすると拒否されます。

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

条件を満たすアーカイブを抽出したPCIeエクストラのルートに、たとえば`model_mlatess_int8.tar.gz`としてコピーし、そのパスを`--model`で渡します。

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model model_mlatess_int8.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

MLAの入出力を直接扱うようにコンパイルされたYOLOv8nアーカイブでは、両方のバージョンがコントラクトと、すべてのヘッドでゼロの偏差を出力します。

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128 range=[0, 1]
  output bbox_0 INT8 [80, 80, 64] scale=0.0828159 zero_point=-60
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 64] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

デフォルトはカード0とキュー0です。別のカードを使用する場合にのみ`--card N`を渡します。

## In Practice

アプリケーションが量子化を所有する場合に`mla_only`を有効にします。センサーや前段のモデルからすでにINT8を生成している場合、独自の後処理のために生のINT8ヘッドが必要な場合、またはカード側のレイテンシからEV74ステージを取り除きたい場合です。すべてのスケール、ゼロポイント、入力範囲は`model.info()`から読み取り、モデルの別のビルドからコピーしないでください。

このルートは全か無かです。複数入力モデルのすべての入力はINT8で到着する必要があり、画像の前処理やボックスデコードを`mla_only`と組み合わせることはできません。ホストがFP32データを保持し、量子化を制御する必要がない場合は、デフォルトルートを使用してください。

デプロイメントの診断については、[PCIeモデルワークフロー](/develop-apps/development-workflow/pcie-model/)に進んでください。

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
