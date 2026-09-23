# 028 外部テンソルメモリをラップする

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15分 |
| Model | yolo_v8s |
| Labels | PCIe, C++, tensor, external memory, zero-copy wrapping |

## Concept

`Tensor::from_external()`を使うと、C++アプリケーションは既存の連続した割り当てを新しいホスト側テンソルへコピーせずにラップできます。テンソルは`std::shared_ptr<void>`の所有者を保持するため、PCIeとGStreamerが使用している間も背後の割り当ては有効です。アプリケーションは、対応する推論結果をプルするまでバイト列を変更してはいけません。

これにより、連続した単一入力に対する追加のホストステージングコピーを回避できます。ただし、エンドツーエンドのゼロコピーではありません。PCIeトランスポートは引き続きペイロードをカード所有のメモリへコピーします。

## Walkthrough

プログラムは再利用可能な3つの入力スロットを作成し、8つの合成FP32フレームを送信します。スロットは、そのスロットに対応する順序付き結果がプルされた後でのみ、利用可能なキューに戻ります。

### モデルのコントラクトを確認する {#step-inspect-contract}

メモリを割り当てる前にモデルを構築し、`info().inputs`を読み取ります。このチュートリアルは単一入力のYOLOv8sアーカイブを使用し、報告されたdtypeがFP32であり、その形状が正確に`size_bytes`バイトになることを確認します。

外部ビューは、対応する`TensorInfo`のdtype、形状、バイトサイズ、名前と一致する必要があります。別のモデルビルドからこれらの値を推測しないでください。

### アプリケーション所有のメモリをラップする {#step-wrap-memory}

各リングスロットは`std::shared_ptr<std::vector<float>>`を所有します。`Tensor::from_external()`には、ベースポインタ、完全なバッキング要素数、共有所有者、モデル形状、ルート名を渡します。ビューが連続しているため、PCIeホストはステージング割り当てを作成せずに直接ラップできます。

生ポインタだけを保持するのでは不十分です。トランスポートは`push()`が戻った後もテンソルを保持できるため、共有所有者が必須です。

### モデルをビルドする {#step-build-model}

モデルコントラクトとリング割り当てを検証した後でビルドします。この例では`max_inflight`をリングサイズに設定し、アプリケーションとトランスポートに同じ明示的な上限を与えます。

### リングを送信して安全に再利用する {#step-submit-ring}

利用可能なスロットを埋め、`push()`を呼び出し、そのスロットを処理中キューに移します。`push()`が戻っただけでストレージを変更または再利用しないでください。利用可能なスロットがない場合、この例は`pull()`を呼び出し、一致する順序付き結果が届いた後でのみ最も古いスロットを利用可能なキューへ戻します。

最後のドレインを含め、受理されたすべてのプッシュに1回のプルを対応させます。タイムアウト時は、まだアクティブかもしれないリクエストのメモリを再利用せず、モデルを閉じます。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているようにチュートリアルバンドルをダウンロードします。抽出したPCIeエクストラのルートへYOLOv8sをダウンロードします。

```bash
sima-cli modelzoo get yolo_v8s
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_028_wrap_external_tensor_memory
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_028_wrap_external_tensor_memory
./build/tutorials-standalone/tutorial_028_wrap_external_tensor_memory
```

デフォルトはカード0とキュー0です。別のカードを使用する場合にのみ`--card N`を渡します。成功すると次のように表示されます。

```text
input=images
ring_slots=3
completed=8
[OK] 028_wrap_external_tensor_memory
```

## In Practice

ホストで直接ラップする経路には連続したストレージが必要です。不連続ストライドのテンソルも記述子が有効なら受け付けられますが、ホストはステージング割り当てへコンパクト化します。個別に割り当てられた複数入力もステージングメモリへパックされます。

複数入力モデルでこのステージング割り当てを避けられるのは、すべてのテンソルが1つの共有パック割り当て内の連続したビューである場合だけです。`info().inputs`が報告する順序で送信し、各入力の名前と形状を使用します。たとえば、両方の入力がFP32の場合は次のようになります。

```cpp
const auto& first = info.inputs.at(0);
const auto& second = info.inputs.at(1);
const std::size_t first_count = first.size_bytes / sizeof(float);
const std::size_t second_count = second.size_bytes / sizeof(float);

auto packed =
    std::make_shared<std::vector<float>>(first_count + second_count);
pcie::Tensor input0 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, first.shape, first.name);
pcie::Tensor input1 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, second.shape, second.name,
    static_cast<std::int64_t>(first.size_bytes));

model.push({input0, input1});
```

この最適化が取り除くのはホスト側のパッキングコピーだけです。PCIeは推論前に、パックされたペイロードをカード所有のトランスポートメモリへ引き続きコピーします。

## ソースファイル

- `run_external_tensor_memory.cpp`
