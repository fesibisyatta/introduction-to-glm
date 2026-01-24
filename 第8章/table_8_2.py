from mlogit import MultinomialLogitOptimizer
import numpy as np
import pandas as pd
import itertools

if __name__ == "__main__":

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

    df_dummies = pd.get_dummies(
        df, 
        columns = ["Sex", "Age"],
        dtype = int
        )

    # 反応の基準カテゴリーは、「重要でない/あまり重要でない」と設定
    # 説明変数の基準カテゴリーは、「女性」、「年齢18-23歳」を設定
    list_exp_var = ["intercept", "Sex_Men", "Age_24-40", "Age_>40"]
    list_target = ["Important", "Very_important"]
    list_coef = []
    for a, b in itertools.product(list_target, list_exp_var):
        list_coef.append(f"{a}_{b}")

    # 説明変数、反応変数、試行回数の取得
    X = df_dummies[list_exp_var[1:]].values
    y = df_dummies[list_target].values

    # n
    n = df_dummies["Total"].values

    # 切片を追加
    X = np.column_stack([np.ones(X.shape[0]), X])
    print(X.shape)
    
    # ロジスティック回帰による推定
    cls = MultinomialLogitOptimizer(X, y, n)
    cls.fit()
    beta, std_error = cls.beta, cls.std_error
    print(" ")
    print(
        pd.DataFrame(
            [beta.flatten() , std_error.flatten()], 
            columns = list_coef,
            index = ["推定値", "標準誤差"]
            ).T
        )
    print(" ")
    log_likelihood, log_likelihood_max, log_likelihood_min = cls._calc_log_likelihood()
    print("最小モデルの最大対数尤度：", log_likelihood_min)
    print("当てはめモデルの最大対数尤度：", log_likelihood)
    print("飽和モデルの最大対数尤度：", log_likelihood_max)
    print(" ")
    print("逸脱度：", cls.calc_deviance())
    print("尤度比カイ二乗統計量：", cls.calc_likelihood_ratio_chi2())
    print("pearsonのカイ二乗統計量 :", cls.calc_chi2_statistic())
    print("擬似R^2値：", cls.calc_pseudo_r_squared())
    print(" ")
    