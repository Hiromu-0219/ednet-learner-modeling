# フェーズ1：行動の組と遷移の集計（監査サンプル KT3・KT4）

作成日：2026-10-22。対象は監査サンプル（KT3 32人・KT4 32人）だけで、数値は設計確認用。研究結果としては報告しない。カテゴリ定義・決定点・閾値はまだ決めていない。

## 結論
- **(action_type, item_type) の組**：KT3 は7種類、KT4 は14種類。KT4 で増えた7種類は、音声・動画の再生と停止、選択肢の消去と取り消し、支払いの行。
- **「ほぼ確定」な状態から出る遷移の割合**（max_p ≥ 0.95、カテゴリで絞らず全行）：KT3 0.177、KT4 0.018。
- **submit 直後の行動**：どちらも解説に入る（enter:e）が最多で、KT3 0.858、KT4 0.724。次は問題に入る（enter:b）で、KT3 0.142、KT4 0.275。
- **KT3 と KT4 の32人**：同一人物はいない（user_id の重なり 0）。**このため、KT3 と KT4 の数値の違いには、データ階層の違いと学習者の違いが混ざっている。**
- **ノートブックとの差分**：ノートブックが読む生データの列は両方にそろっている。違うのは、列名以外の想定（item_type `p`、例の `CATEGORY_MAP` にない組）と、CLAUDE.md の作業ノートブック名。

## 条件
- データ：`data/audit/KT3`・`data/audit/KT4`（抽出条件は `data_provenance.md`）。
- 読み込みと並べ替え：ノートブックのセル4・5と同じ。SEED=0 で32ファイルを選び（32ファイルしかないので全員）、`user_id`→`timestamp` で安定ソート。同一時刻の行は元ファイルの順を保つ。
- `item_type`：`item_id` の先頭1文字。
- 遷移：同じ学習者の中の、次の1行との組。ノートブックと同じく、セッション（60分）の区切りをまたぐ遷移も含む。
- **ノートブックとの違い**：セクション3のセル13は `CATEGORY_MAP` で割り当てた行だけを対象にするが、カテゴリ定義は未決定なので、全行を対象にした。ノートブックそのものは書き換えておらず、同じ計算を別スクリプトで実行した（numpy 2.4.6、pandas 3.0.6）。
- エントロピーの単位は bit。

## 1. (action_type, item_type) の組と件数

### KT3（14,102行）
| action_type | item_type | 件数 | 割合 |
|---|---|---|---|
| respond | q | 3,788 | 0.269 |
| submit | b | 2,678 | 0.190 |
| enter | b | 2,678 | 0.190 |
| quit | e | 2,468 | 0.175 |
| enter | e | 2,468 | 0.175 |
| enter | l | 11 | 0.001 |
| quit | l | 11 | 0.001 |

### KT4（6,524行）
| action_type | item_type | 件数 | 割合 |
|---|---|---|---|
| respond | q | 1,373 | 0.210 |
| enter | b | 1,125 | 0.172 |
| submit | b | 1,125 | 0.172 |
| quit | e | 855 | 0.131 |
| enter | e | 855 | 0.131 |
| pause_audio | b | 391 | 0.060 |
| play_audio | b | 387 | 0.059 |
| erase_choice | q | 179 | 0.027 |
| play_video | l | 91 | 0.014 |
| pause_video | l | 91 | 0.014 |
| enter | l | 24 | 0.004 |
| quit | l | 24 | 0.004 |
| pay | p | 3 | 0.000 |
| undo_erase_choice | q | 1 | 0.000 |

- どちらでも、enter:b と submit:b、enter:e と quit:e、enter:l と quit:l の件数が一致する。

## 2. トークン単位の遷移（ノートブック セクション3）

### KT3（遷移 14,070）
「ほぼ確定」な状態から出る遷移の割合：**0.177**

| token | n | max_p | argmax_next | entropy_bits | ほぼ確定 |
|---|---|---|---|---|---|
| respond:q | 3,788 | 0.648 | submit:b | 0.939 | |
| enter:b | 2,678 | 0.917 | respond:q | 0.414 | |
| submit:b | 2,653 | 0.858 | enter:e | 0.593 | |
| enter:e | 2,468 | 1.000 | quit:e | 0.005 | ✓ |
| quit:e | 2,461 | 0.918 | enter:b | 0.433 | |
| enter:l | 11 | 1.000 | quit:l | 0.000 | ✓ |
| quit:l | 11 | 1.000 | enter:b | 0.000 | ✓ |

submit の直後（2,653件）：enter:e 2,276（0.858）/ enter:b 376（0.142）/ enter:l 1（0.000）

### KT4（遷移 6,492）
「ほぼ確定」な状態から出る遷移の割合：**0.018**

| token | n | max_p | argmax_next | entropy_bits | ほぼ確定 |
|---|---|---|---|---|---|
| respond:q | 1,373 | 0.665 | submit:b | 1.305 | |
| enter:b | 1,125 | 0.719 | respond:q | 0.972 | |
| submit:b | 1,098 | 0.724 | enter:e | 0.859 | |
| enter:e | 855 | 0.935 | quit:e | 0.357 | |
| quit:e | 850 | 0.915 | enter:b | 0.476 | |
| pause_audio:b | 391 | 0.512 | submit:b | 1.926 | |
| play_audio:b | 387 | 0.571 | pause_audio:b | 1.508 | |
| erase_choice:q | 179 | 0.380 | respond:q | 1.851 | |
| play_video:l | 91 | 1.000 | pause_video:l | 0.000 | ✓ |
| pause_video:l | 91 | 0.736 | play_video:l | 0.832 | |
| enter:l | 24 | 1.000 | play_video:l | 0.000 | ✓ |
| quit:l | 24 | 0.583 | enter:b | 0.980 | |
| pay:p | 3 | 0.667 | enter:b | 0.918 | |
| undo_erase_choice:q | 1 | 1.000 | respond:q | 0.000 | ✓ |

submit の直後（1,098件）：enter:e 795（0.724）/ enter:b 302（0.275）/ enter:l 1（0.001）

- 次の行と timestamp が同じ遷移の割合：KT3 0.001、KT4 0.014。KT4 では、同一時刻の行をどの順に並べるかで一部の遷移が変わる。

## 3. KT3 と KT4 の32人は同一人物か
- user_id の重なりは **0人**。KT3 の32人は `manifest.json` の `audit_users` と一致した。
- KT4 の32人は KT4 全体から SEED=0 で新しく抽出した人たち（`data_provenance.md`）。

## 4. ノートブックの想定との差分（書き換えていない）

| 箇所 | ノートブックの想定 | 実データ | 修正案 |
|---|---|---|---|
| 生データの列（セル4〜6、18、22） | `timestamp, action_type, item_id, source, platform` を参照 | KT3・KT4 とも、すべてある。KT4 には加えて `cursor_time` があるが、ノートブックは使っていない | 不要 |
| セル6のコメント | `item_id` の先頭文字は q・b・e・l | KT4 には `p`（pay、3行）もある | コメントに `p=支払い` を足す |
| セル9 `CATEGORY_MAP` の例 | 8組（うち `play_audio:b` はコメントアウト） | KT4 の14組のうち、例にない組が6つある：`pause_audio:b`、`erase_choice:q`、`play_video:l`、`pause_video:l`、`pay:p`、`undo_erase_choice:q`。`play_audio:b` もコメントアウトされているので、未割り当てになる | セクション2でカテゴリを決めるときに、14組すべてを割り当てる（対象外は `None`） |
| セル13 | `CATEGORY_MAP` で割り当てた行だけで遷移を数える | カテゴリ定義が未決定なので、そのままでは計算できない | 今回は全行で計算した。カテゴリ決定後にノートブックで再実行する |
| CLAUDE.md | 作業ノートブックは `phase1_ednet_audit.ipynb` | KT4 に確定したので、KT4 の作業は `phase1_ednet_audit_KT4.ipynb` | CLAUDE.md の作業ノートブック名を書き換えるか決める |

## 解釈と注意（推測を含む）
- KT4 の「ほぼ確定」の割合が KT3 より低い（0.177→0.018）のは、主に enter:e→quit:e が KT4 では 0.935 で 0.95 を下回ったためで、閾値の近くにある。KT3 でも enter:b（0.917）や quit:e（0.918）は閾値のすぐ下にある。**この割合は閾値 0.95 の選び方に敏感なので、決定点の定義では閾値の感度も見たほうがよい。**
- KT3 と KT4 の差は学習者の違いも含む。データ階層だけの違いを見たい場合は、KT3 監査32人の KT4 ファイル（`data/raw/KT4/` にある）で同じ集計をすると比べられる。
- submit 直後の行動は、どちらも enter:e か enter:b にほぼ限られる。決定点の候補としては「submit の直後に解説を見るか、次の問題に進むか」が残る。ただ、解説が自動で開くのか学習者が選んでいるのかは、ログからは分からない。
