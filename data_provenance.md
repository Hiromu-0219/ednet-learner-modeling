# データの取得記録

生データは `data/` に置き、git では管理しない（`.gitignore`）。このファイルに取得元・ハッシュ・抽出条件を記録する。

## EdNet-KT4
- 配置：`data/raw/EdNet-KT4.zip`、展開先 `data/raw/KT4/`
- ZIP サイズ：1,169,540,359 bytes
- ZIP の SHA-256：`ae7b699f64980a2e795e3a7f02b144f32f822ba0a5b925826db98f2ef195fd40`
- 配置日：2026-10-21
- 取得元：公式 GitHub（riiid/ednet）の README にあるダウンロードリンク（2026-10-21 本人より）
- 学習者ファイル数：297,915（`u*.csv`）
- カラム：`timestamp, action_type, item_id, cursor_time, source, user_answer, platform`

## EdNet-KT3（全体）
- 配置：`data/raw/EdNet-KT3.zip`、展開先 `data/raw/KT3/`
- ZIP サイズ：762,774,456 bytes（learner-simulation の記録にある公式配布物のサイズと一致）
- 配置日：2026-10-21
- 取得元：公式 GitHub（riiid/ednet）の README にあるダウンロードリンク（2026-10-21 本人より）
- ZIP の SHA-256：`cc7823d9a5a0570c81b2c279178547b27ca5c628209f37648f0707484f8efa78`
- 学習者ファイル数：297,915（`u*.csv`）
- カラム：`timestamp, action_type, item_id, source, user_answer, platform`
- `data/raw/__MACOSX/` は展開時にできた macOS のメタデータで、データではない

## 監査サンプル（KT3、32人）
- 配置：`data/audit/KT3/`（`u*.csv` 32ファイルと `manifest.json`）。2026-10-21 に `data/audit/` 直下から移動
- コピー元：`Desktop/learner-simulation/data/raw/phase2_sample/KT3`（2026-10-21 にコピー、SHA-256 一致を確認）
- 抽出条件：`SHA256("learner-simulation-phase2-audit-v1:" + user_id)` の昇順で先頭32人（`manifest.json` の `selection`）
- KT3 全体の同名ファイルと32人すべてがバイト単位で一致することを確認（2026-10-21）
- 32人すべての KT4 ファイルが `data/raw/KT4/` にあることを確認（2026-10-21）
- この32人は本番の評価に使わない

## 監査サンプル（KT4、32人）
- 配置：`data/audit/KT4/`（`u*.csv` 32ファイルと `manifest.json`）
- 抽出条件：`files = sorted(Path("data/raw/KT4").glob("u*.csv"))` から `np.random.default_rng(0).choice(files, size=32, replace=False)`。ノートブックのセル2・4と同じ規則（numpy 2.4.6、2026-10-21）
- 母集団：297,915ファイル（除外なし）
- KT3 監査32人との重複：なし。監査ユーザーは KT3・KT4 合わせて64人
- 各ファイルの SHA-256・バイト数・行数と抽出された順は `manifest.json` に記録
- この32人も本番の評価に使わない
