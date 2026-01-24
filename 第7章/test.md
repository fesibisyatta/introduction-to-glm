---
title: "多項ロジスティック回帰の係数推定"
emoji: "💨"
type: "idea"
topics:
  - "一般化線形モデル"
  - "dobson"
  - "irls"
published: true
published_at: "2026-01-11 01:30"
---

# 問題設定
共変量パターンが $N$ 個, 反応変数のカテゴリーが $J$ 個, 説明変数は切片項を含めて $p + 1$ 個ある状況を考える. このとき, カテゴリー間に自然な順序がない場合における2項ロジスティック回帰モデルの拡張を考える. 

$Y_{ij}$ を第 $i$ 共変量パターンにおけるカテゴリー $j$ の反応変数,$\,$ $\mathbb{x}_i = (x_{i0}, x_{i1}, \cdots, x_{ip})^T$ を第 $i$ 共変量パターン,$\,$ 基準カテゴリーをカテゴリー1とし, $j \ge 2$ に対してカテゴリー $j$ に対する回帰係数を $\beta_j = (\beta_{0j}, \beta_{1j}, \cdots, \beta_{pj})$ をとする. ただし, $i = 1, \cdots, N, \, j = 1, \cdots, J$, $x_{i0} = 1$ である. このとき

$$
\begin{aligned}
logit(\pi_{ij}) = \log \left( \frac{\pi_{ij}}{\pi_{i1}} \right) = \mathbb{x}_i^T \mathbb{\beta}_{j}
\end{aligned}
$$

として定式化できる. また, 上式と $\sum_{j = 1}^J \pi_{ij} = 1$ を組み合わせることで 

$$
\begin{aligned}
\pi_{i1} &= \frac{1}{1 + \sum_{j = 2}^J \exp(\mathbb{x_i}^T \mathbb{\beta}_{j})} \\

\pi_{ij} &= \frac{\exp(\mathbb{x_i}^T \mathbb{\beta}_{j})}{1 + \sum_{j = 2}^J \exp(\mathbb{x_i}^T \mathbb{\beta}_{j})} \\
\end{aligned}
$$

を得る.


# 対数尤度関数の導出
本文p164 ~ p165より, 多項分布は第 $i$ 共変量パターンの度数 $n_i$ で条件付けしたポアソン変数と見做せるので, 尤度関数は

$$
\begin{aligned}
L(\mathbb{\beta}_1, \cdots, \mathbb{\beta}_J; n_1, \cdots, n_J) = \prod_{i = 1}^N \frac{n_i!}{\prod_{j = 1}^J y_{ij}!} \prod_{j = 1}^J \pi_{ij}^{y_{ij}}
\end{aligned}
$$

となり, 対数尤度関数は

$$
\begin{aligned}
l(\mathbb{\beta}_1, \cdots, \mathbb{\beta}_J; n_1, \cdots, n_J) &= \log \left( \prod_{i = 1}^N \frac{n_i!}{\prod_{j = 1}^J y_{ij}!} \prod_{j = 1}^J \pi_{ij}^{y_{ij}} \right) \\

&= \sum_{i = 1}^N \left(\log \frac{n_i!}{\prod_{k = 1}^J y_{ik}!} + \sum_{j = 1}^J y_{ij} \log \pi_{ij} \right) \\

&= \sum_{i = 1}^N \left(\log n_i ! - \sum_{k = 1}^J \log y_{ik}! + \sum_{j = 1}^J y_{ij} \log \pi_{ij} \right) \\

&= C + \sum_{i = 1}^N \left(y_{i1} \log \pi_{i1} + \sum_{j = 2}^J y_{ij} \log \pi_{ij} \right) \\

&= C + \sum_{i = 1}^N \left\{ -y_{i1} \log (1 + \sum_{j = 2}^J \exp(\mathbb{x}_i^T \mathbb{\beta}_j))  \right\} \\

&+ \sum_{i = 1}^N  \left\{ \sum_{j = 2}^J y_{ij} \log \left(\frac{\exp(\mathbb{x}_i^T \mathbb{\beta}_j)}{1 + \sum_{k = 2}^J \exp(\mathbb{x}_i^T \mathbb{\beta}_k)} \right)   \right\} \\

&= C + \sum_{i = 1}^N \left\{ \sum_{j = 2}^J y_{ij} \mathbb{x}_i^T \mathbb{\beta}_j - \sum_{j = 1}^J y_{ij} \left(\log(1 + \sum_{k = 2}^J \exp(\mathbb{x}_i^T \mathbb{\beta}_k)) \right) \right\} \\

&= C + \sum_{i = 1}^N \left\{ \sum_{j = 2}^J y_{ij} \mathbb{x}_i^T \mathbb{\beta}_j - n_i \left(\log(1 + \sum_{k = 2}^J \exp(\mathbb{x}_i^T \mathbb{\beta}_k)) \right) \right\}
\end{aligned}
$$

となる. ただし, $\pi_{ij}$ に無関係な変数は $C$ としてまとめ, 最後の等式は, $n_i = \sum_{j = 1}^J y_{ij}$ に注意して展開した.


# 対数尤度関数の偏微分
$2 \le j \le J, 0 \le l \le p$ とする.

$$
\begin{aligned}
U_{lj} &= \frac{\partial}{\partial \beta_{lj}}l(\mathbb{\beta}_1, \cdots, \mathbb{\beta}_J; n_1, \cdots, n_J) \\
&= \sum_{i = 1}^N \left(x_{il} y_{ij} - n_i \frac{x_{il} \exp(\mathbb{x}_i^T \mathbb{\beta}_j)}{1 + \sum_{k = 2}^J \exp(\mathbb{x}_i^T \mathbb{\beta}_k)} \right) \\
&= \sum_{i = 1}^N x_{il} \left(y_{ij} - n_i \pi_{ij}\right)
\end{aligned}
$$

となる. なお, 切片項のため, $x_{i0} = 1$ であることに注意する.

# 分散共分散行列の導出
$2 \le j \le J, 0 \le l \le p$ とする. 各共変量パターン $i$ ごとに反応変数は独立だと仮定しているので, $p \neq q$ のとき

$$
E[Y_{pj}Y_{qk}] = E[Y_{pj}]E[Y_{qk}]
$$

を満たす. また, $Y_{ij}$ は多項分布に従っていることから, $j \neq k$ のとき, 共分散を求めると

$$
\operatorname{Cov}(Y_{ij}, Y_{ik}) = - n_i \pi_{ij} \pi_{ik}
$$

となる.[^共分散] 故に

$$
\begin{aligned}
\sum_{p \neq q} x_{pl} x_{qs} E[(Y_{pj} - n_p \pi_{pj})(Y_{qk} - n_q \pi_{qk}) = 0
\end{aligned}
$$

である. このことを用いると

$$
\begin{aligned}
E[U_{lj} U_{sk}] &= E \left[ \left( \sum_{i = 1}^N x_{il} (Y_{ij} - n_i \pi_{ij}) \right) \left( \sum_{i = 1}^N x_{is} (Y_{ik} - n_i \pi_{ik}) \right) \right] \\
&= \sum_{i = 1}^N x_{il}x_{is} E[(Y_{ij} - n_i \pi_{ij}) (Y_{ik} - n_i \pi_{ik})] \\
&+ \sum_{p \neq q} x_{pl} x_{qs} E[(Y_{pj} - n_p \pi_{pj})(Y_{qk} - n_q \pi_{qk})] \\
&= \sum_{i = 1}^N x_{il}x_{is} E[(Y_{ij} - n_i \pi_{ij}) (Y_{ik} - n_i \pi_{ik})] \\

&=
\begin{cases}
\sum_{i = 1}^N n_i x_{il} x_{is} \pi_{ij} (1 - \pi_{ij}), & j = k, \\[8pt]
\sum_{i = 1}^N x_{il} x_{is} \operatorname{Cov}(Y_{ij}, Y_{ik}), & j \neq k. \\[8pt]
\end{cases} \\

&=
\begin{cases}
\sum_{i = 1}^N n_i x_{il} x_{is} \pi_{ij} (1 - \pi_{ij}), & j = k, \\[8pt]
- \sum_{i = 1}^N n_i \pi_{ij} \pi_{ik} x_{il} x_{is} , & j \neq k.
\end{cases} \\[8pt]
\end{aligned} \\
$$

となる.[^行列による表示]

# 回帰係数の更新式
以上より, スコアベクトル $\mathbb{U}$ および情報行列 $\mathfrak{J}$ をそれぞれ

$$
\mathbb{U} = 
\begin{pmatrix}
U_{02} \\
\vdots \\
U_{p2} \\
\vdots \\
U_{0J} \\
\vdots \\
U_{pJ} \\
\end{pmatrix}
$$

$$
\mathfrak{J} = 
\begin{pmatrix}
E[U_{02}^2] & \cdots & E[U_{02} U_{p2}] & \cdots & E[U_{02} U_{0J}] & \cdots & E[U_{02} U_{pJ}] \\
\vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
E[U_{p2}U_{02}] & \cdots & E[U_{p2}^2] & \cdots & E[U_{p2} U_{0J}] & \cdots & E[U_{p2} U_{pJ}] \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
E[U_{0J}U_{02}] & \cdots & E[U_{0J}U_{p2}] & \cdots & E[U_{0J}^2] & \cdots & E[U_{0J} U_{pJ}] \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
E[U_{pJ}U_{02}] & \cdots & E[U_{pJ}U_{p2}] & \cdots & E[U_{pJ} U_{0J}] & \cdots & E[U_{pJ}^2 ] \\
\end{pmatrix}
$$

と置く. 更新式を次で定義する.

$$
\begin{aligned}
\mathbb{b}^{(m)} = \mathbb{b}^{(m - 1)} + \mathfrak{J^{(m - 1)}}^{-1} \mathbb{U}^{(m - 1)}
\end{aligned}
$$

# 数値計算（Dobson本 例8.3.1 ： 自動車装備に対する嗜好調査）
ここでは, これまで導出した内容を踏まえ, https://www.kyoritsu-pub.co.jp/book/b10010684.html の例8.3.1にある自動車装備に対する嗜好調査を多項ロジスティック回帰で分析することを考える. 

多項ロジスティック回帰を以下で実装する.[^オーバーフローの防止][^初期値]

```python
class MultinomialLogitOptimizer:
    
    def __init__(self, X, y, n):
        """
        X: 説明変数の行列 (N x (p+1))  ※0列目は1（切片）
        y: 各カテゴリの反応回数 (N x (J-1)) ※基準カテゴリを除いたもの
        n: 各共変量パターンの総試行回数 (N)
        """
        self.X = X
        self.y = y
        self.n = n
                
        # 係数ベクトル b の初期化 ( (p+1)*(J-1) 次元)
        self.beta = np.zeros((self.y.shape[1], self.X.shape[1]))

    # カテゴリー分類確率
    def pi_ij(self):
        
        # 線形予測子 z の計算
        z = self.X @ self.beta.T
        
        # 行ごとの最大値を差し引いてオーバーフローを防止
        # keepdims=True により (n_samples, 1) となり、ブロードキャストが可能
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        
        # 分母の計算 (参照カテゴリの 1 を含める)
        # 1に対しても exp(-z_max) を掛けることで数学的整合性を保つ
        denominator = np.exp(-z_max) + exp_z.sum(axis=1, keepdims=True)
        
        return exp_z / denominator
    
    # スコアおよび情報行列の計算
    def compute_score_and_information_matrix(self):
                
        # スコアの計算
        U_lj = self.X.T @ (self.y - self.n.reshape(self.X.shape[0], 1) * self.pi_ij())    
        U = np.concatenate([U_lj[:, 0], U_lj[:, 1]])
    
        J = []
        for j in range(U_lj.shape[1]):
            for l in range(U_lj.shape[0]):
                for k in range(U_lj.shape[1]):
                    for s in range(U_lj.shape[0]):
                        if j == k:
                            w = (self.n * self.X[:, l] * self.X[:, s] * self.pi_ij()[:, j] * (1 - self.pi_ij()[:, j])).sum()
                        else:
                            w = (-1)*(self.n * self.pi_ij()[:, j] * self.pi_ij()[:, k] * self.X[:, l] * self.X[:, s]).sum()   
                        J.append(w)
    
        # 情報行列の計算
        J = np.array(J).reshape((U_lj.shape[0]*  U_lj.shape[1], U_lj.shape[0] *  U_lj.shape[1]))

        return U, J

    # IRLSによる更新
    def fit(self, n_iters = 10):

        for iter_ in range(n_iters):

            U, J = self.compute_score_and_information_matrix()
    
            # betaの更新
            beta_new = self.beta.flatten() + np.linalg.inv(J) @ U
            
            # 更新前後の誤差
            delta = ((beta_new - self.beta.flatten())**2).sum()
            self.beta = beta_new.reshape((y.shape[1], X.shape[1]))        
            
            # betaの更新幅
            if delta < 1e-10:
                break
            
        return self.beta, np.sqrt(np.diag(np.linalg.inv(J)))

```

また, 例8.3.1のデータセットを用意し, 説明変数および反応変数に前処理を施す. なお, 本書の結果を再現するため, 反応変数の基準カテゴリーは「重要でない/あまり重要でない」, 説明変数の基準カテゴリーは「Women」「Age_18-23」とした. 

```python
# 表8.1のデータ
data = [
    # Women
    ["Women", "18-23", 26, 12, 7, 45],
    ["Women", "24-40", 9, 21, 15, 45],
    ["Women", ">40",   5, 14, 41, 60],

    # Men
    ["Men", "18-23", 40, 17, 8, 65],
    ["Men", "24-40", 17, 15, 12, 44],
    ["Men", ">40",   8, 15, 18, 41],
]

columns = [
    "Sex",
    "Age",
    "No_or_little_importance",
    "Important",
    "Very_important",
    "Total"
]

df = pd.DataFrame(data, columns=columns)

df_dummies = pd.get_dummies(df, columns = ["Sex", "Age"], dtype = int)

# 反応の基準カテゴリーは、「重要でない/あまり重要でない」と設定
# 説明変数の基準カテゴリーは、「女性」、「年齢18-23歳」を設定
X = df_dummies[["Sex_Men", "Age_24-40", "Age_>40"]].values
y = df_dummies[["Important", "Very_important"]].values

# n
n = df_dummies["Total"].values

# 切片を追加
X = np.column_stack([np.ones(X.shape[0]), X])
```

以下のコードで10回IRLSによる推定を行う.
```python
cls = MultinomialLogitOptimizer(X, y, n)
beta, std_error = cls.fit()
```

`beta`の値は以下になる.
```
array([[-0.59079879, -0.38812814,  1.12826576,  1.58770717],
       [-1.03907572, -0.81301751,  1.47810691,  2.91675142]])
```
また, その標準誤差`std_error`は
```
array([0.28397548, 0.30051129, 0.3416449 , 0.40289909, 0.33050162,
       0.32103815, 0.40092566, 0.42292744])
```
である. これらをmarkdownでまとめると

| 係数   | 推定値        | 標準誤差      |
|-------|---------------|---------------|
| $\beta_{02}$ : 定数項 | -0.59079879   | 0.28397548    |
| $\beta_{12}$ : 男性 | -0.38812814   | 0.30051129    |
| $\beta_{22}$ : 24-40歳 |  1.12826576   | 0.34164490    |
| $\beta_{32}$ : >40歳 |  1.58770717   | 0.40289909    |
| $\beta_{03}$ : 定数項  | -1.03907572   | 0.33050162    |
| $\beta_{13}$ : 男性 | -0.81301751   | 0.32103815    |
| $\beta_{23}$ : 24-40歳 |  1.47810691   | 0.40092566    |
| $\beta_{33}$ : >40歳 |  2.91675142   | 0.42292744    |

となる. 本文p170の表8.2と見比べるとほぼ係数や標準誤差は一致している. 推定自体はうまくいっているように見える.

![](https://storage.googleapis.com/zenn-user-upload/ba61738537fe-20260111.png)
*Dobson著： 一般化線形モデル入門 第2版 例8.3.1の表8.2*

以上で, 多項ロジスティック回帰モデルの導出とPythonによる数値計算が完了した.

# 最後に
Dobson本の第4章に推定方法の記載があるが, あくまで反応変数が2値であったり連続変数の場合にしか適用できない（ように見える）ため, 多値分類にも拡張できるようにした. この実装を行う上で悩んだのが

- Fisherによるscoring methodでは, 対数尤度関数のhessianを計算していたが, dobsonのIRLSではスコアの分散共分散行列で計算を実施しており, 推定方法として同一か
- 多クラス分類を実行するため, カテゴリー分類確率を $\pi_i$ から $\pi_{ij}$ への拡張

の2点である. 前者は, Information Matrix Equality という性質を用いることで, 等価であるということがわかっている.[^IME] 後者については, $\pi_i$としてではなく, $\pi_{ij}$ としてカテゴリーも明示的に考慮することで解決した. 2値分類は実質的なカテゴリーは1つ（一方の分類確率が決まればもう一方も決まるため）で良いので楽であったが, 多値分類の場合はそうはいかなかったので苦労した.

なお, この記事とは若干異なる方法ではあるが, 多項ロジスティック回帰の推定方法について解説している和書として

https://www.kspub.co.jp/book/detail/5178027.html

が挙げられる. この本では

- 共変量パターンを集計データではなく, グループ化される前のデータに対して適用している. つまり, 任意の$i$に対して, $n_i$ = 1である. また, $Y_i$ は0/1のフラグ変数である.
- betaの分散として, スコアの分散共分散行列ではなく, 対数尤度関数のhessianの期待値を採用している.

の2点が異なる. 結果自体はどちらで定式化しても一致するので, もし興味がある方がいればそちらを読んでいただくのも良いかもしれない.

[^共分散]: 多項分布の共分散の導出はbellcurveの[この記事](https://bellcurve.jp/statistics/course/26597.html?srsltid=AfmBOoprXkfg9ltk6M6xUrQKLR6c81Taeotc1QD9VMF35VCTaPL1SlOY)などを参照

[^行列による表示]:行列を使って綺麗に表現できるような気がするのだが, わからんかったので, このまま愚直に計算することにする.

[^オーバーフローの防止]: https://www.anarchive-beta.com/entry/2020/06/08/180000 などで解説されているが, softmax関数は発散してしまい, 浮動小数点の許容範囲を超えてしまうことがある. しかし, 指数関数の肩から"最大値"を引くことで, 発散を抑えることができる.

[^初期値]: $beta$ の初期値を0にしている. その他の初期値にすると収束しなかったり, 逆行列が特異になったりと, 不安定であった. 推定方法の問題なのかは不明.

[^IME]: Information Matrix Equalityについては  https://alecospapadopoulos.wordpress.com/wp-content/uploads/2014/05/the-information-matrix-equality1.pdf などを参考にした.