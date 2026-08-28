---
title: "貢献する"
description: "Neatフレームワークのアーキテクチャ、ビルド、テスト、およびリリース作業に関するコントリビューター向けガイド。"
sidebar_position: 1
slug: /develop-apps/contribute/
---

# 貢献

このセクションは、Neat Library 自体を変更するコントリビューター向けです。リポジトリの構造、変更のビルドとテストの方法、およびアプリケーション開発者が維持する必要がある契約について説明します。

<div class="overview-section-label">貢献者向けオリエンテーション</div>

このリポジトリにある **Neat Library** は、C++/Pythonライブラリとランタイムです。
モデルアーカイブをロードし、パイプラインを構成および検証し、Modalix ハードウェア上で実行し、パブリックなアプリケーションAPIを公開します。Neat SDK と DevKit Sync は、それらを取り巻く開発ワークフローです。

このリポジトリを変更する際には、人間とエージェントによる開発の両方を支援する、フレームワークの特性を最適化してください。具体的には、明確なAPI、決定的な動作、構造化された診断、厳格な検証、および安定したパブリックコントラクトです。

<div class="overview-section-label">貢献者向けガイドライン</div>

- **決定性こそが重要** — 要素名、生成されたパイプライン、レポート、およびテストを再現可能に保つ。
- **デバッグの容易性を最優先** — 失敗時には、文字列だけでなく、構造化されたデータを出力する。
- **無言のフォールバックは行わない** — モデル入力のバグやハードウェア/ランタイムの障害を隠蔽しない。
- **実行前に検証する** — 構造、上限、形状、および契約に関するエラーをランタイム前に検出する。
- **公開 API は安定性を維持する** — `include/*` にインストールされたヘッダーには、追加的で互換性のある変更のみを適用する。
- **並行処理は制限され、監視可能でなければならない** — クリーンアップ処理がハングアップせず、診断がスレッドセーフでなければならない。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>ここから始めましょう。</h2>
    <p>コードを変更する前に、リポジトリ、命名規則、および期待される動作について理解しておいてください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>アーキテクチャ</strong><span>リポジトリの構造、モジュールの所有権、ランタイムのフロー、および拡張ポイントについて学びましょう。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>契約書の名称</strong><span>一貫して、標準化された製品名、API名、パッケージ名、および型名を使用してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>コーディング規約</strong><span>C++のスタイル、公開API、互換性、およびドキュメントに関する規定に従ってください。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>構築とテスト</h2>
    <p>フレームワークを構築し、変更を検証し、Pythonバインディングの作業を行います。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>構築する</strong><span>ソースコードから Neat をビルドし、CMake プロファイルまたはビルドオプションを選択します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>テスト要件</strong><span>各変更の種類に対して、どのようなテストとドキュメントの更新が必要になるかを把握しておきましょう。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Pythonバインディング</strong><span>コントリビューターとして、PyNeat のバインディングを構築、テスト、パッケージ化します。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>契約と内部構造</h2>
    <p>モデルアーカイブ、プラグインコントラクト、または内部パッケージを変更する際にこれらを使用してください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>MPK契約</strong><span>モデルアーカイブの取り込み、検証、およびセキュリティルールを理解してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>プラグイン JSON 真偽マップ</strong><span>フリーズされた SIMA プラグインの JSON 契約マップを確認してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/appcomplex_workspace_packaging/"><strong>AppComplex パッケージ</strong><span>ゲート付きのアプリケーション複合ワークスペースサービスパッケージを構築およびインストールします。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>リリースとメンテナンス</h2>
    <p>これらは、リリース時のチェックポイント、クリーンアップ計画、および長期にわたる設計指針として使用してください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>リリースチェックリスト</strong><span>リリースを妨げる条件、必要な確認事項、および再現可能な手順に従ってください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>エラー分類の展開</strong><span>構造化されたエラーコードの移行と検証の状況を追跡します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>エージェントによるワークフロー</strong><span>APIが、人間とAIによる共同開発を支援するように設計されている理由をご覧ください。</span></a></li>
    </ul>
  </section>
</div>
