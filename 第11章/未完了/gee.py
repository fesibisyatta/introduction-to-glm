import numpy as np
import pandas as pd


def calc_std_error(D, V_inv, S, is_sandwitch):
    if is_sandwitch:
        # 情報行列の計算
        J = np.zeros((2, 2))
        for i in range(len(D)):
            J += D[i].T @ V_inv[i] @ D[i]

        # Cの計算
        C = np.zeros((2, 2))
        for i in range(len(D)):
            C += D[i].T @ V_inv[i] @ np.outer(S[i], S[i]) @ V_inv[i] @ D[i]

        # サンドイッチ推定量
        std_error = np.linalg.inv(J) @ C @ np.linalg.inv(J)

    else:
        std_error = np.zeros((2, 2))
        for i in range(len(D)):
            std_error += D[i].T @ V_inv[i] @ D[i]
        std_error = np.linalg.inv(std_error)

    return std_error


def compute_matrices(df_result, beta, R, A, psi, unbaised_variance):
    """_summary_

    Args:
        df_result (_type_): _description_
        beta (_type_): _description_
        R (_type_): _description_
        A (_type_): _description_
        psi (_type_): _description_
        unbaised_variance (_type_): _description_

    Returns:
        _type_: _description_
    """

    D = []
    V_inv = []
    S = []
    pearson_resid = []
    p = 2  # 特徴量の数

    # グループごとに計算
    for i, (_, group) in enumerate(df_result.groupby("dog")):
        # muの勾配
        Di = np.stack([np.ones(len(group)), group["x"].values], axis=1)
        D.append(Di)

        # yiの推定
        # 実際には link function の逆関数を噛ませる必要があるが、
        # リンク関数として恒等関数を選択したのでそのままの値
        mu_i_hat = (Di * beta).sum(axis=1)

        # 残差
        resid = group["y"].values - mu_i_hat
        S.append(resid)

        # 正規分布の分散
        # 不偏分散を毎回iterごとに更新したものにするのも良いかも
        if unbaised_variance:
            variance = (resid**2).sum() / (8 - p)
        else:
            variance = 1

        # ピアソン残差
        ri = resid / np.sqrt(variance)
        pearson_resid.append(ri)

        # 分散の推定
        V_i = np.sqrt(A[i]) @ R[i] @ np.sqrt(A[i]) * psi

        # 逆行列を格納
        V_inv_i = np.linalg.inv(V_i)
        V_inv.append(V_inv_i)

    return D, V_inv, S, pearson_resid


def update_beta(D, V_inv, S, beta, pearson_resid, df_result, p=2):
    """_summary_

    Args:
        D (_type_): _description_
        V_inv (_type_): _description_
        S (_type_): _description_
        beta (_type_): _description_
        pearson_resid (_type_): _description_
        df_result (_type_): _description_
        p (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    # subjectの数だけ足し合わせる
    mat_1 = np.zeros((2, 2))
    mat_2 = np.zeros((2, 1))
    for i in range(len(D)):
        mat_1 += D[i].T @ V_inv[i] @ D[i]
        mat_2 += D[i].T @ V_inv[i] @ S[i].reshape(8, 1)

    # betaの更新
    beta += (np.linalg.inv(mat_1) @ mat_2).reshape(-1)

    # 作業相関行列Rおよびスケーリングパラメータpsiの更新
    R = [None for _ in range(len(df_result["dog"].unique()))]
    A = [None for _ in range(len(df_result["dog"].unique()))]
    psi = sum((pearson_resid[i] ** 2).sum() for i in range(len(pearson_resid))) / (
        len(df_result) - p
    )
    for i in range(len(pearson_resid)):
        Ri = np.outer(pearson_resid[i], pearson_resid[i]) / (len(df_result) - p)
        R[i] = Ri
        A[i] = np.diag(np.diag(Ri))

    return beta, R, A, psi


def irls_for_gee(df_result, is_sandwitch, unbaised_variance):
    df_result = df_result.copy()

    # 特徴量の数
    p = 2

    # 繰り返し回数
    n_iters = 5

    # 反復計算
    for iter_ in range(n_iters):
        # 初期値の設定
        if iter_ == 0:
            # 作業相関行列およびAの初期値は単位行列とする
            beta = np.array([1.0, 1.0])
            R = [np.eye(8) for _ in range(df_result["dog"].nunique())]
            A = [np.eye(8) for _ in range(df_result["dog"].nunique())]
            psi = 1

        else:
            pass  # 2回目以降は前回の値を使用

        # 各種行列の計算
        D, V_inv, S, pearson_resid = compute_matrices(
            df_result, beta, R, A, psi, unbaised_variance
        )

        # beta, R, A, psiの更新
        beta, R, A, psi = update_beta(D, V_inv, S, beta, pearson_resid, df_result, p)

    # 最終的な標準誤差の計算
    std_error = calc_std_error(D, V_inv, S, is_sandwitch)

    return beta, np.diag(std_error)


if __name__ == "__main__":
    df_result = pd.read_csv("dog_data_long_format.csv")
    beta, std_error = irls_for_gee(df_result, is_sandwitch=True, unbaised_variance=True)
    print("Beta estimates:", beta)
    print("Standard Errors:", std_error)
