---
title: "高度な概念"
description: "より高度な Neat アプリケーションを構築するための設計の詳細"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# 高度な概念

アプリケーションで、基本的な`Model`、`Graph`、および`Run`のワークフロー以上の機能が必要な場合は、これらのページを参照してください。これらのページでは、より高度なNeatアプリケーションの背後にある契約とランタイムの動作について説明します。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>アプリケーション設計</h2>
    <p>アプリケーションの構成方法と、その出力方法を設計します。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>グラフ</strong><span>モデル、ノード、名前付きの入力と出力、分岐、および実行を構成します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>MIPIカメラを使用してください。</strong><span>libcameraの入力とCVUによる前処理を使用して、ソースが所有するカメラからモデルへのグラフを構築します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>ビデオを送信</strong><span>Neat アプリケーションから、H.264 RTP/UDP を使用してビデオ出力をストリーミングします。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>JSONメタデータを送信します。</strong><span>構造化されたアプリケーションのメタデータを UDP JSON 形式で公開します。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>実行モデル</h2>
    <p>作業がどのようにスケジュールされ、スレッド化され、パイプライン層にマッピングされるかを理解してください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>タイミングモデル</strong><span>同期処理と非同期処理、プッシュ／プル、そして処理がいつ行われるかを理解しましょう。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>スレッドモデル</strong><span>どのスレッドが存在し、アプリケーションコードがどこで実行される可能性があるかを把握しておきましょう。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>GStreamer レイヤー</strong><span>Neatがどのような処理を抽象化しているのか、また、生のGStreamerの詳細がいつ重要になるのかを学びましょう。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>データとモデルに関する契約</h2>
    <p>パイプラインが依存するテンソル、メモリ、モデルの契約について理解しましょう。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>データ形式</strong><span>マップ形式のトークンを、テンソルのレイアウト、形状、およびプレーンのセマンティクスにマッピングします。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>データ型コントラクト</strong><span>前処理、MLA、後処理の各段階におけるテンソルの精度がどのように変化するかを確認してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>メモリモデル</strong><span>ゼロコピーバッファー、物理アドレス、およびキャッシュの動作について理解する。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>モデル ランタイム</h2>
    <p>コンパイルされたモデルのアーティファクトや、それらを実行するハードウェアバックエンドについて、さらに詳しく見ていきましょう。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>MPK契約</strong><span>コンパイルされたモデルアーカイブが、推論の段階と契約をどのように定義するかを確認してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>プロセッサのバックエンド</strong><span>A65、CVU、MLA、MLASHM、APU、TVM、およびM4の役割を理解してください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>CVUカーネル</strong><span>グラフ構築の際の、前処理および後処理の各段階を再確認してください。</span></a></li>
    </ul>
  </section>
</div>
