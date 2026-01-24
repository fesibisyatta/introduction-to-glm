import numpy as np
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
            self.beta = beta_new.reshape((self.y.shape[1], self.X.shape[1]))        
            
            # betaの更新幅
            if delta < 1e-10:
                break
            
        return self.beta, np.sqrt(np.diag(np.linalg.inv(J)))
