# 009 NumPy配列をモデルに渡す

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | numpy, pytorch, tensor, io |

## Concept

Neatのテンソルと、既存のデータ構造（NumPy配列、PyTorchテンソル、または`cv::Mat`）の間でデータを移動させ、レイアウト、データ型、およびコピーのセマンティクスを制御することで、既存の推論スタックにNeatを組み込むことができます。

## Walkthrough

Neatを既存の推論スタックに統合する場合、これが連携の境界となります。ホストデータがNeatの`Tensor`に、そしてNeatの`Tensor`が再びホストデータになる方法です。これを最初から正しく行うことで、レイアウトの誤り、暗黙のデータ型変換、2つの環境間の予期しないエイリアシングなど、一般的な統合時のバグを防ぐことができます。

ここが、2つの言語が最も異なる点でもあります。PythonユーザーはNumPy/PyTorchを、C++ユーザーはOpenCVを使用します。変換の*概念*は同じですが、API名と型が異なるため、以下に示す言語ごとの説明が重要になります。最終的には、ホストデータをNeatのテンソルに変換し、コピーせずにペイロードを検査し、ソースバッファが削除されても安全な、所有権のあるコピーを作成できるようになります。

### ホストデータをテンソルとしてラップ {#step-to-tensor}

最初のステップでは、すでに保持しているデータをNeatの`Tensor`に変換します。画像のレイアウトを明示的にタグ付けします(`RGB`)。これにより、ランタイムがバイトを正しく解釈し、推測しなくなります。`copy=True`（またはC++のCPUメモリの選択）は、テンソルが自身のバイトを所有するか、ソースとエイリアスするかを決定します。ソースバッファが変更または解放される可能性がある場合は、明示的な所有権が安全なデフォルトです。

**C++:** `simaai::neat::from_cv_mat(mat, ImageSpec::PixelFormat::RGB, TensorMemory::CPU)`は、`cv::Mat`をCPUベースのテンソルにラップします。

**Python:** `pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)`は、HWC `uint8` NumPy配列をラップします。

### ペイロードを検査 {#step-map-and-inspect}

データがテンソルになったら、それを読み戻すことができます。これは、連携の半分の処理です。下流にデータを渡す前に、形状とバイトが変換によって保持されていることを確認します。

**C++:** `tensor.map_read()`は、生の`data`ポインタと`size_bytes`を公開する`Mapping`を返します。これはテンソルのストレージへの*ビュー*であり、コピーは行われません。そのため、例では先頭のバイトを直接チェックサムできます。

**Python:** `tensor.to_numpy(copy=True)`は、テンソルからNumPy配列を生成します。例では、HWCレイアウトが正しく変換されたことを確認するために、その`.shape`を出力します。

### コピーを所有する {#step-own-a-copy}

最後に、元のソースバッファから完全に切り離されたデータを作成します。これにより、入力が削除された後も安全にデータを保持できます。これは、長期間使用するコンシューマに渡すコピーです。

**C++:** `tensor.clone()`は、それが由来する`cv::Mat`とは独立した、新しいCPU所有のストレージにコピーします。

**Python:** PyTorch を使用して、同様の概念を示します。`pyneat.Tensor.from_torch(t, copy=True, ...)` と `tensor.to_torch(copy=True)` は、所有されている PyTorch テンソルを介して往復処理を行います。（`torch` がインストールされていない場合は、正常にスキップされます。）

## Run

**Neat インストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** コマンドを実行します。**ソースからビルド** コマンドは、**リポジトリのルート**から実行します。この章では、モデルアーカイブは必要ありません。

**Python:**
```bash
python3 share/sima-neat/tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py \
  --width 128 --height 96
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_009_pass_numpy_to_model
./build/tutorials-standalone/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

予想される出力（C++）：

```text
tensor_rank=3
tensor_bytes=36864
head_checksum=4342
clone_bytes=36864
[OK] 009_pass_numpy_to_model
```

予想される出力（Python、`torch` がインストールされている場合）：

```text
numpy_roundtrip_shape=(96, 128, 3)
torch_roundtrip_shape=(96, 128, 3)
```

（`torch` がない場合、Python ビルドは torch 行の代わりに `torch_roundtrip_skipped=True` を出力します。）この章の C++ ソースを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## In Practice

インターオペラビリティの概要を、簡単な参照のために、往復デモを過ぎた後に示します。

### 変換 API

- NumPy: `pyneat.Tensor.from_numpy(array, copy=..., image_format=...)`（入力）、`tensor.to_numpy(copy=...)`（出力）。
- PyTorch: `pyneat.Tensor.from_torch(tensor, copy=..., image_format=...)`（入力）、`tensor.to_torch(copy=...)`（出力）。
- OpenCV (C++): `simaai::neat::from_cv_mat(mat, pixel_format, memory)`（入力）、ゼロコピービューの場合は `tensor.map_read()`、所有権のあるコピーの場合は `tensor.clone()`。

### コピーとビュー

- `copy=True`（Python）/ `clone()`（C++）は、ソースから切り離されたデータを提供します。ソースが解放または変更された後でも、安全に使用できます。
- `copy=False` / `map_read()` は、ソースを参照するビューを提供します。コストは低いですが、ソースが存続し、変更されていない場合にのみ有効です。

### レイアウトと dtype

- 画像データの場合は、常に明示的な `image_format` / `PixelFormat` を渡して、レイアウトを解釈させ、推測させないようにします。
- Neat は、dtype を自動的に変換しません。テンソルの dtype をモデルの入力コントラクトに一致させてから、モデルに渡します。

## ソースファイル
- C++: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.cpp`
- Python: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py`
