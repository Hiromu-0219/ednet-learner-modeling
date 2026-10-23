# source 列の意味と件数：自分で選んだ行動と推薦された行動を分けられるか（監査サンプル、KT4）

作成日：2026-10-23。集計：`src/source_audit.py`。対象は監査サンプル（KT4 監査32人、KT4 同じ32人）だけ。

## 結論
- **公式の説明で、分け方の手がかりは得られる。**
  - 自分で選ぶもの：`sprint`（学ぶパートを選ぶ）、`archive`（講義を選ぶ）。
  - 推薦・提案されたもの：`todays_recommendation`・`tutor`・`adaptive_offer`。
- **ただし、実データの値と公式の説明が一部合わない。**
  - 監査サンプルに `todays_recommendation`・`in_review`・`after_sprint` は1件もなかった。
  - 逆に、`review`・`review_quiz`・`diagnosis` は説明がないか、説明と名前が違う。
- **はっきり分けられる行動はごく少ない。**
  - 自分で選んだことがはっきりしている `archive`・`my_note` と、推薦がはっきりしている `adaptive_offer`・`tutor` は、合わせても全行の 1.4〜11.8%。使った学習者も1〜5人しかいない。
  - 全行の 74〜95% は `sprint` だった。`sprint` は「パートは自分で選ぶが、その中の問題はアプリが出す」もので、どちらとも言い切れない。
- **監査サンプルでは、自発性の分析ができるかは判断できない。** 件数を見る限り、本番の規模（多くの学習者）で `archive`・`my_note`・`adaptive_offer` を使う人がどのくらいいるかを確かめる必要がある。

## 1. 公式の説明（riiid/ednet の README）
| source | 公式の説明（要約） | 自分で選んだか |
|---|---|---|
| `sprint` | 学ぶパートを学習者が選び、そのパートの問題だけを解く | **一部**：パートは自分で選ぶ。問題そのものはアプリが出す（と読める）【要確認】 |
| `todays_recommendation`（`::sprint`、`::review_quiz`、`::lecture`） | 予測した正答確率にもとづき、毎日問題と講義を推薦 | 推薦 |
| `adaptive_offer` | 特定のタグで不正解が一定数を超えたとき（または正答率が下がったとき）に、講義と問題を提案 | 推薦（提案） |
| `tutor` | 「All Parts」で、アルゴリズムが全候補から選んだ問題 | 推薦 |
| `in_review` | 復習の仕組みで、解いた問題をやり直す | 自分で（復習を始める） |
| `archive` | すべての講義にアクセスでき、そこから講義を見る | **自分で選ぶ** |
| `my_note` | 解いた問題の解説を読み直す | **自分で選ぶ** |
| `after_sprint`・`after_review` | sprint・review の問題の解説 | 問題の source に従う |
| `diagnosis` | 説明なし（例の表に出てくるだけ） | 実力診断と推測。アプリが出す |

- 出典：[riiid/ednet README](https://github.com/riiid/ednet/blob/master/README.md) の KT2・KT3 の節。
- README には「講義を見る経路は2つ」とあるが、実際には3つ（`archive`、`adaptive_offer`、`todays_recommendatin::lecture`）が挙げられている。綴りの誤り（recommendatin）も原文のまま。

## 2. 監査サンプルでの件数

### KT4 監査32人（SEED=0、6,524行）
| source | 行 | 割合 | 学習者 | 入った先（enter：問題 / 解説 / 講義） |
|---|---|---|---|---|
| sprint | 4,813 | 0.738 | 13 | 802 / 751 / 0 |
| diagnosis | 889 | 0.136 | 30 | 233 / 0 / 0 |
| my_note | 317 | 0.049 | 3 | 44 / 56 / 0 |
| adaptive_offer | 308 | 0.047 | 5 | 38 / 36 / 11 |
| archive | 146 | 0.022 | 2 | 0 / 0 / 13 |
| review_quiz | 35 | 0.005 | 1 | 7 / 7 / 0 |
| review | 10 | 0.002 | 1 | 0 / 5 / 0 |
| tutor | 3 | 0.000 | 1 | 1 / 0 / 0 |
| （欠損） | 3 | 0.000 | 3 | pay の行 |

### KT4 同じ32人（25,875行）
| source | 行 | 割合 | 学習者 | 入った先（enter：問題 / 解説 / 講義） |
|---|---|---|---|---|
| sprint | 24,585 | 0.950 | 10 | 2,405 / 2,406 / 0 |
| diagnosis | 817 | 0.032 | 32 | 228 / 0 / 0 |
| adaptive_offer | 329 | 0.013 | 4 | 32 / 30 / 11 |
| review_quiz | 75 | 0.003 | 2 | 11 / 10 / 0 |
| my_note | 28 | 0.001 | 4 | 0 / 14 / 0 |
| review | 27 | 0.001 | 1 | 0 / 6 / 0 |
| tutor | 12 | 0.000 | 1 | 2 / 2 / 0 |
| （欠損） | 2 | 0.000 | 2 | pay の行 |

- 講義に入った行は、`archive`（自分で選ぶ）と `adaptive_offer`（提案）からだけだった。
- 解説は、問題と同じ source（主に `sprint`）で記録されていた。README の `after_sprint` という値は使われていなかった。

## 3. 分け方の案（決めていない）
| 区分 | source | 根拠 |
|---|---|---|
| 自分で選んだ（はっきり） | `archive`、`my_note`、`review` | 学習者が自分から講義・解説・復習を開く（README） |
| 推薦・提案された | `adaptive_offer`、`tutor`、`todays_recommendation*`、`review_quiz`（README では todays_recommendation の一種）【要確認】 | アプリが選んで出す |
| 一部だけ自分で選んだ | `sprint` | パートは選ぶが、問題はアプリが出す【要確認】 |
| 対象外 | `diagnosis` | 実力診断。流れが固定（D2） |

## 4. 注意
- 監査サンプルは64人だけで、はっきり分けられる source を使った学習者は1〜5人しかいない。この件数では、行動の違いは比べられない。
- `sprint` の中で、問題を学習者が選べるのか、アプリが出すのかは、README からははっきり読み取れない。【要確認：EdNet の論文本文、Santa の画面の説明】
- `review` と `review_quiz` の違い、`diagnosis` の意味は、README に説明がない。
- 「自分で選んだ」は行動の記録の区別であって、「興味を持った」ことを示すものではない。
