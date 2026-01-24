import numpy as np
import math
from scipy.special import gammaln

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
                            w = (
                                self.n * self.X[:, l] * self.X[:, s] * self.pi_ij()[:, j] * (1 - self.pi_ij()[:, j])
                                ).sum()
                        else:
                            w = (-1)*(
                                self.n * self.pi_ij()[:, j] * self.pi_ij()[:, k] * self.X[:, l] * self.X[:, s]
                                ).sum()   
                        J.append(w)
    
        # 情報行列の計算
        J = np.array(J).reshape((U_lj.shape[0]*  U_lj.shape[1], U_lj.shape[0] *  U_lj.shape[1]))

        return U, J
    
    def compute_score_and_information_matrix_kronecker(self):
    
        # スコアの計算
        U_lj = self.X.T @ (self.y - self.n.reshape(self.X.shape[0], 1) * self.pi_ij())    
        U = np.concatenate([U_lj[:, 0], U_lj[:, 1]])

        # π行列の取得
        pi = self.pi_ij()
        n_categories = pi.shape[1]
        
        # カテゴリ間の共分散行列を計算
        # Σ[j,k] = E[δ_jk - π_k | π_j] = δ_jk π_j - π_j π_k
        Sigma = np.zeros((self.X.shape[0], n_categories, n_categories))
        for j in range(n_categories):
            for k in range(n_categories):
                if j == k:
                    Sigma[:, j, k] = pi[:, j] * (1 - pi[:, j])
                else:
                    Sigma[:, j, k] = -pi[:, j] * pi[:, k]
        
        # 各サンプルの情報行列を計算してから合計
        # J = Σ_i n_i * (Σ_i ⊗ x_i x_i^T)
        J = np.zeros((n_categories * self.X.shape[1], n_categories * self.X.shape[1]))
        for i in range(self.X.shape[0]):
            x_outer = np.outer(self.X[i], self.X[i])  # x_i x_i^T
            # クロネッカー積: Σ_i ⊗ x_i x_i^T
            J += self.n[i] * np.kron(Sigma[i], x_outer)
        
        return U, J

    # IRLSによる更新
    def fit(self, n_iters = 10):

        for iter_ in range(n_iters):

            U, J = self.compute_score_and_information_matrix()
    
            # betaの更新
            beta_new = self.beta.flatten() + np.linalg.inv(J) @ U
            
            # 更新前後の誤差
            delta = ((beta_new - self.beta.flatten())**2).sum()
            self.beta = beta_new.reshape((self.y.shape[1], self.X.shape[1]))        
            
            # betaの更新幅
            if delta < 1e-10:
                break
        
        self.J = J
        self.std_error = np.sqrt(np.diag(np.linalg.inv(J)))
    
    def predict_y(self):
        """
        予測値の計算
        """
        pi = self.pi_ij()
        y_pred = self.n.reshape(self.X.shape[0], 1) * pi
        return y_pred

    def _calc_log_likelihood(self):
        """
        対数尤度と最大対数尤度の計算
        """

        def __log_likelihood_without_comb(y, n, pi):
            # y * log(pi) の計算 (y=0 のとき 0*log(0)=0 とする)
            item1 = np.where(y > 0, y * np.log(pi), 0).sum()
            
            # (n - y_sum) * log(1 - pi_sum) の計算
            y_sum = y.sum(axis=1, keepdims=True)
            pi_sum = pi.sum(axis=1, keepdims=True)
            n_minus_y = n - y_sum
            
            # n - y_sum = 0 のとき 0*log(0)=0 とする
            item2 = np.where(
                n_minus_y > 0, 
                n_minus_y * np.log(1 - pi_sum), 
                0
                ).sum()
            
            return item1 + item2

        pi = self.pi_ij()                       # 当てはめモデルの確率
        n = self.n.reshape(len(self.X), 1)      # (I, 1)
        y = self.y                              # (I, J)

        # 組み合わせ項
        log_comb = (
            gammaln(n.flatten() + 1) 
            - (
                gammaln(y + 1).sum(axis = 1) 
                + gammaln(n.flatten() - y.sum(axis = 1) + 1)
                )
            ).sum()

        # 当てはめモデルの最大対数尤度
        log_comb = 0 # 組み合わせ項は定数なので, 例8.3.1では省略
        log_likelihood = __log_likelihood_without_comb(y, n, pi) + log_comb

        # 最大モデル（飽和モデル）最大対数尤度
        pi_max = y / n
        log_likelihood_max = __log_likelihood_without_comb(y, n, pi_max) + log_comb

        # 最小モデルの最大対数尤度
        total_y = y.sum(axis=0)  # 各カテゴリの合計
        total_n = n.sum()
        pi_min = total_y / total_n  # (J,)
        pi_min = np.tile(pi_min, (y.shape[0], 1))  # (I, J)
        log_likelihood_min = __log_likelihood_without_comb(y, n, pi_min) + log_comb

        return log_likelihood, log_likelihood_max, log_likelihood_min

    def calc_deviance(self):
        """
        逸脱度の計算
        """
        log_likelihood, log_likelihood_max, _  = self._calc_log_likelihood()

        return 2 * (log_likelihood_max - log_likelihood)
    
    def calc_likelihood_ratio_chi2(self):
        """
        尤度比カイ二乗統計量の計算
        """
        log_likelihood, _, log_likelihood_min = self._calc_log_likelihood()

        return 2 * (log_likelihood - log_likelihood_min)

    def _pearson_chi_sq_resid(self):
        pi = self.pi_ij()  # (I, J)
        n = self.n.reshape(self.X.shape[0], 1)
        
        # 基準カテゴリの確率を追加
        pi_base = 1 - pi.sum(axis=1, keepdims=True)  # (I, 1)
        pi_all = np.hstack([pi, pi_base])  # (I, J+1)
        
        # 基準カテゴリの観測度数を追加
        y_base = n - self.y.sum(axis=1, keepdims=True)  # (I, 1)
        y_all = np.hstack([self.y, y_base])  # (I, J+1)
        
        # 期待度数
        y_pred = n * pi_all
        
        # Pearson残差 (期待度数が0の場合を考慮)
        resid = np.where(y_pred > 0, 
                        (y_all - y_pred) / np.sqrt(y_pred), 
                        0)
        return resid
        
    def calc_chi2_statistic(self):
        """
        Pearsonのカイ二乗統計量の計算
        """
        resid = self._pearson_chi_sq_resid()
        chi2_stat = (resid**2).sum()
        return chi2_stat
    
    def calc_pseudo_r_squared(self):
        """
        疑似R二乗の計算
        """
        log_likelihood, _, log_likelihood_min = self._calc_log_likelihood()
        return 1 - (log_likelihood / log_likelihood_min)