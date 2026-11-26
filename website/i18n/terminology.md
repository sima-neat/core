# Localization terminology

This glossary is normative for Korean, Japanese, Traditional Chinese, and Ukrainian Neat
documentation. Product names and source-code identifiers remain unchanged.

| English | Korean (`ko`) | Japanese (`ja`) | Traditional Chinese (`zh-Hant-TW`) | Ukrainian (`uk-UA`) |
| --- | --- | --- | --- | --- |
| Palette SDK | Palette SDK | Palette SDK | Palette SDK | Palette SDK |
| Neat Library | Neat Library | Neat Library | Neat Library | Neat Library |
| Neat Examples | Neat Examples | Neat Examples | Neat Examples | Neat Examples |
| Modalix | Modalix | Modalix | Modalix | Modalix |
| DevKit | DevKit | DevKit | DevKit | DevKit |
| PyNeat | PyNeat | PyNeat | PyNeat | PyNeat |
| Insight | Insight | Insight | Insight | Insight |
| LLiMa | LLiMa | LLiMa | LLiMa | LLiMa |
| Hugging Face | Hugging Face | Hugging Face | Hugging Face | Hugging Face |
| Model Compiler | Model Compiler | Model Compiler | Model Compiler | Model Compiler |
| Model Zoo | Model Zoo | Model Zoo | Model Zoo | Model Zoo |
| graph | 그래프 | グラフ | 圖 | граф |
| graph fragment | 그래프 조각 | グラフフラグメント | Graph 片段 | фрагмент графа |
| pipeline | 파이프라인 | パイプライン | 管線 | конвеєр |
| runtime | 런타임 | ランタイム | 執行階段 | середовище виконання |
| model archive | 모델 아카이브 | モデルアーカイブ | 模型封存檔 | архів моделі |
| artifact | 아티팩트 | アーティファクト | 成品 | артефакт |
| host | 호스트 | ホスト | 主機 | хост |
| deploy / deployment | 배포 | デプロイ | 部署 | розгортання |
| compile / compilation | 컴파일 | コンパイル | 編譯 | компіляція |
| validate / validation | 검증 | 検証 | 驗證 | перевірка |
| troubleshooting | 문제 해결 | トラブルシューティング | 疑難排解 | усунення несправностей |
| release notes | 릴리스 노트 | リリースノート | 版本資訊 | примітки до випуску |
| memory | 메모리 | メモリ | 記憶體 | пам’ять |
| thread | 스레드 | スレッド | 執行緒 | потік |
| tensor (concept) | 텐서 | テンソル | 張量 | тензор |
| architecture | 아키텍처 | アーキテクチャ | 架構 | архітектура |
| application (software) | 애플리케이션 | アプリケーション | 應用程式 | застосунок |
| building block | 구성 요소 | 構成要素 | 建構模組 | складова |
| folder | 폴더 | フォルダー | 資料夾 | тека |
| source of truth | 단일 진실 공급원 | 信頼できる唯一の情報源 | 權威來源 | джерело істини |
| plugin (concept) | 플러그인 | プラグイン | 外掛程式 | плагін |
| configuration / configure | 구성 / 설정 | 構成 / 設定 | 設定 | конфігурація / налаштування |
| build (software) | 빌드 | ビルド | 建置 | збірка |
| load | 로드 | 読み込み | 載入 | завантаження |
| data | 데이터 | データ | 資料 | дані |
| file | 파일 | ファイル | 檔案 | файл |
| user | 사용자 | ユーザー | 使用者 | користувач |
| return (from an API) | 반환 | 返す | 傳回 | повернення |

## Ukrainian term choices

These Ukrainian choices were agreed in the native review of the Ukrainian
documentation (#911). They apply to Ukrainian only. The last column lists
variants not to use for that meaning. Those words stay correct in their other
meanings, for example «вміст» for content, «типовий» for typical, «каталог»
for a catalog, «Опис» for a Description column, «вихідний» for output
(вихідні дані, вихідний тензор), and «пакет» for a software package or a batch.

| English | Ukrainian (`uk-UA`) | Avoid for this meaning |
| --- | --- | --- |
| payload (concept) | корисні дані | вміст, корисне навантаження |
| default | за замовчуванням | типовий |
| detection (object detection) | виявлення | детекція |
| directory | тека | каталог |
| Meaning (table column header) | Значення | Опис |
| source code / source file | початковий код / файл початкового коду | вихідний код, вихідний файл |
| floating point | рухома кома (число з рухомою комою) | плаваюча кома, плаваюча точка |
| throw / raise (an exception) | генерувати виняток | викидати виняток, кидати виняток |
| source language (ASR, translation) | мова оригіналу | вихідна мова, мова джерела |
| bundle (distributed set of files: tutorial, offline, Core, Scheduler bundle) | набір | пакет, комплект |
| bundle (`Sample` kind `Bundle`) | зразок типу `Bundle`; in prose набір (іменованих полів) | пакет, пакетний зразок |
| PCIe Co-Processing (category name) | Співобробка через PCIe | Співобробка PCIe, Співпроцесинг PCIe, спільна обробка через PCIe |
| head (model output: class head, packed heads) | вихід (виходи моделі), вихідний тензор | голова, головка, хедер, хед |

## Protected text

Keep the following text exactly as authored:

- API symbols and type names, including `Graph`, `Run`, `Tensor`, and `Sample`
- commands, command output, filenames, paths, environment variables, and error messages
- fenced code blocks and `ShellCommand` components
- URL targets, anchors, JSX/MDX imports, and structural component names

Translate a generic concept naturally, but retain its English spelling when it
is an API symbol. For example, translate “a tensor contains data,” but do not
translate the `Tensor` type.
