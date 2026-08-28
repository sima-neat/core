# 010 讀取並解讀模型輸出

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | output, patterns, sink |

## Concept

從 `run.pull()` 或 `model.run()` 安全地讀取。每次執行都會傳回一個 `Sample`，這是一種小型總和類型，可能是一個張量、一組具名欄位，或兩者皆有，因此您在處理有效載荷之前，應先對其進行分類。

## Walkthrough

在您最佳化吞吐量或新增複雜圖邏輯之前，您需要一種穩定且可靠的方法來讀取執行階段傳回的任何內容。輸出始終是一個 `Sample`，但其形狀會有所不同：它可能是一個單一的張量，或者是一組帶有名稱的欄位（如第 009 章所述）。嘗試從一組欄位中獲取 `.tensor`，或者假設不存在的形狀，這就是本章要教您避免的錯誤。

我們建置與之前相同的最小同步圖，執行一個框架，然後有系統地*檢查*結果：其 `kind`，是否存在張量，有多少個欄位，以及張量的秩。到最後，您將擁有一個可重複使用的輸出讀取模式，適用於執行階段服務的任何模型。

### 設定輸入 {#step-configure-input}

宣告輸入合約——像素 `format`、`width`、`height`、`depth`——與我們將推送的框架匹配。這是整個章節中使用的相同邊界合約。

### 組合並建置圖 {#step-compose-graph}

將輸入節點連接到輸出節點，並 `build()` 成一個同步 `Run`，傳遞框架，以便 `build()` 可以協商具體的形狀。由於中間沒有模型，因此輸出反映了輸入——這正是使其成為研究輸出結構的理想場所的原因。

### 執行一個框架 {#step-run-frame}

推送一個框架並同步提取一個結果。單個 `run(...)` 呼叫是單框架快捷方式；它傳回的內容是我們在此要剖析的對象。

**C++：** `run.run(...)` 傳回一個 `TensorList`——對於單張量輸出，這意味著一個條目，下一步將通過 `out.size()` 和 `out.front()` 檢查該條目。

**Python：** `run.run(...)` 傳回一個 `Sample`，直接公開 `.kind`、`.tensor`、`.tensors` 和 `.fields`。

### 檢查樣本 {#step-inspect-sample}

這就是本節的重點：在讀取有效負載之前，先讀取結構。首先檢查存在性和類型，然後從張量的 `shape` 中推導出秩。保護每個步驟（非空輸出、非空形狀）是使輸出讀取器能夠可靠地處理您無法控制其形狀的模型的原因。

**C++：** 報告 `out.size()` 和張量的存在性，如果為空或 `out.front().shape` 為空，則拋出異常，然後從 `shape.size()` 中列印 `rank`。（`fields=0` 行是一個佔位符——`TensorList` 不會攜帶 Python `Sample` 攜帶的欄位結構。）

**Python：** 輸出 `sample.kind`、`sample.tensor is not None`、`len(sample.fields)`，以及第一個張量的等級——將完整的總和類型表面整合在一個地方。對於張量類型的結果，存在 `.tensor`；對於張量集合的結果，讀取 `.tensors`；對於捆綁分支，讀取 `.kind` 和 `.fields`。

## Run

從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預建版本）** 命令；從 **程式碼庫根目錄**執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/011_interpret_model_output/interpret_model_output.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_interpret_model_output
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_interpret_model_output
./build/tutorials-standalone/tutorial_011_interpret_model_output
```

預期輸出（C++）：

```text
outputs=1 has_tensor=yes fields=0
rank=3
[OK] 011_interpret_model_output
```

Python 建置透過 `Sample` 表面輸出相同的資訊：

```text
sample_kind=SampleKind.TensorSet
has_tensor=False
num_fields=0
output_rank=3
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

用於讀取任何模型的輸出的防禦性檢查清單。

### 在讀取之前進行分類

- 首先檢查 `kind`。單個張量結果為 `SampleKind.Tensor`；多欄位結果為 `SampleKind.Bundle`。
- 對於張量類型，存在 `tensor`，且 `fields` 為空。對於捆綁類型，讀取 `fields`，不要假設存在 `tensor`。

### 驗證合約

- 在取消引用張量之前，確認張量存在。
- 在計算等級或索引維度之前，確認 `shape` 不為空。
- 當消費者預期特定的元素類型時，檢查 `tensor.dtype`。

## 原始程式碼檔案
- C++：`tutorials/011_interpret_model_output/interpret_model_output.cpp`
- Python：`tutorials/011_interpret_model_output/interpret_model_output.py`
