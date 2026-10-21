---
title: 硬體測試擁有權
description: 執行加速器測試時，不將逾時視為 DMA 已安全結束
---

# 硬體測試擁有權

測試期限是判定失敗的期限，不是終止 DMA 擁有者的許可。
硬體驗證會在 DevKit 上使用 `tools/hardware_test_supervisor.py`。
常駐監督程式擁有實際的加速器程序。CTest、SSH 與 Python 客戶端僅擁有回報連線。

## 設定與准入

以 `NEAT_GUARDED_HARDWARE_TESTS=ON` 建置 Core，或使用
`SIMANEAT_GUARDED_HARDWARE_TESTS=ON` 設定 CMake。原生 CTest 目標會成為回報代理；
測試引數、環境、標籤、資源鎖與現有的 `TIMEOUT` 屬性維持不變。
已安裝的額外元件提供相同的監督程式客戶端。效能矩陣需要
`SIMA_HARDWARE_SUPERVISOR_SOCKET`。`SIMA_PERF_HOST_FIXTURES_ONLY=1` 僅供非 AArch64
測試固定資料使用，無法啟用未受保護的 AArch64 矩陣執行。

具權限的驗證擁有者提供全新、私用且僅使用一次的狀態資料夾與設定。
監督程式會拒絕舊的准入／失敗停止資料夾。設定會固定下列項目：

- `identity`：`admission_id`、`boot_id`、`cohort_sha256`、`lease_pid`、
  `lease_start_ticks`、`observer_id`。
- `state_dir`、該目錄內的 `socket`，以及 `supervisor_sha256`。
- `checker`：由 root 擁有的確切唯讀檢查器 `command`、執行檔 `sha256`，
  以及不占用資源的探查 `timeout_sec`。
- `executables`：一般檔案的絕對路徑與 SHA256 身分、確切的 `arguments`、
  `deadline_sec` 與受保護的 `required_env` 值。多種核准的命令／期限組合應放在 `variants`。

僅在原始測試沒有逾時限制時使用 `deadline_sec: null`。CTest 代理會要求確切核准的
期限；矩陣則要求其未變更的 180 秒期限。拒絕載入器注入，並在 `required_env` 中固定
標準執行階段／外掛程式路徑，以及數值驗證基準設定。

檢查器會收到 `--phase launch|active|retired --owner-json PATH`。它必須傳回新的 JSON，
內含 `schema: 1`、`status: "ready"`、確切固定的 `identity` 與 `checked_at_epoch`。
它絕不可啟動加速器工作或變更硬體。

擁有者記錄綁定 PID、啟動 tick、開機 ID、工作階段 ID、執行 ID 及完整要求的 SHA256。
在允許 `exec` 前，監督程式會將擁有權持久記錄，並在尚未占用資源的子程序等待於
啟動屏障時，重新檢查准入與客戶端存活狀態。運作中的迴圈會記錄工作階段成員、
驗證身分、檢查即時准入狀態，並掃描持久日誌中的未安全結束／隔離狀態。

實際以非負狀態碼結束後，`exit.json` 會綁定擁有者、傳回碼、期限、要求雜湊與觀察到的
完成時間。`retired` 檢查器必須加入：

- `retirement.proven: true`；
- `kind: "authoritative-driver-retirement"` 或 `"reviewed-graceful-exit"`；
- 非空的證據 `token`；
- `scope: "whole-owner-process-tree-and-dma"`；
- `owner`：確切的擁有者記錄。

檢查器實作必須以已審查的執行與傳輸完成契約、實際擁有權／子孫程序檢查、
最新的故障觀察，以及目前的驅動程式／韌體／套件組身分，證實上述證據。
僅取樣到零個工作、程序結束、時間經過或 SSH 連線正常都不是證明。
純 CPU 測試固定資料可使用經專門審查的安全結束契約；不要為其虛構驅動程式證據。

## 逾時與復原

監督程式在逾時、斷線、取消、無法確認安全結束或觀察器失敗時，絕不傳送 TERM/KILL。
它會封存不可逆的失敗停止狀態，保留擁有者與持久日誌，並拒絕後續啟動。
矩陣保留 `ENV_BROKEN / ENV_TIMEOUT`，並將剩餘情境記為**未執行**；
絕不捏造通過或可接受的略過。延遲到達的輸出無法撤銷逾時。
自然結束且已完全安全釋放的數值失敗仍是失敗案例，不會與擁有權不明混為一談。

輸出檔案的完成條件不代表已安全釋放。選用的應用程式停止要求需要確切核准的
完成條件／引數組合，以及固定的處理常式／排空審查憑證。僅允許這種明確審查過的
SIGTERM 要求；它絕不是清理動作，也絕不升級為 SIGKILL。
驗收前仍必須有實際結束與權威性的安全釋放證據。

不要停止 remoteproc、卸載／解除綁定驅動程式、刪除借用緩衝區，或終止狀態不明的
擁有者。請使用裝置現有且受保護的復原程序。復原後建立新的准入；舊准入絕不重新開放。
任何暫時韌體試驗結束後，都必須還原並驗證原始韌體、服務與復原原則。

## 客戶端與證據

Python 客戶端使用 `run_owned(command, cwd, env, deadline_sec, evidence_dir,
socket_path, completion=None, graceful_stop=None)`。CLI 支援
`run-json --request FILE --result FILE --socket PATH`。結果會標示
`completed`、`timeout`、`fail_stop` 或 `refused`、擁有者身分、實際傳回碼、
安全釋放證明及持久 stdout/stderr 路徑。客戶端結束無法終止實際的硬體子程序；
常駐程式必須在目標上獨立執行，而非作為會受工作階段清理影響的 SSH 前景程序。

在硬體准入前先執行無害的原生主機測試。驗證報告必須區分：在此已審查安全框架下
使用未變更的工作負載／驗證基準／期限，以及執行未修改的舊框架或實際託管的 CI 工作流程。

對於 Python 3.13 及更新版本，PyNeat 中繼資料選擇 NumPy `>=2.1,<3`；較舊 Python
則選擇 `>=1.24,<2`。中繼資料解析不代表 ndarray 數值等價。
驗證該 Python/NumPy 套件組前，請先執行已安裝目標的 dtype、步幅、複製／零複製與生命週期測試。
