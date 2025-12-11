import numpy as np
import pandas as pd
import statsmodels.api as sm

N_CLUSTERS = 3
CLUSTER_SIZE = 10


def make_dataset(
    n_clusters: int = N_CLUSTERS, cluster_size: int = CLUSTER_SIZE
) -> pd.DataFrame:
    """_summary_

    Returns:
        _type_: _description_
    """
    np.random.seed(123)

    N = n_clusters * cluster_size

    cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

    # 1つの説明変数
    x = np.random.normal(size=N)

    # クラスタ単位のランダム効果 + 個別誤差
    cluster_effect = np.random.normal(scale=1.0, size=n_clusters)
    y = (
        1.0
        + 2.0 * x
        + cluster_effect[cluster_ids]
        + np.random.normal(scale=0.3, size=N)
    )

    df = pd.DataFrame({"y": y, "x": x, "cluster": cluster_ids})

    return df


if __name__ == "__main__":
    # データセットの作成
    df = make_dataset(n_clusters=N_CLUSTERS, cluster_size=CLUSTER_SIZE)

    # 混合効果モデルの適用
    X = sm.add_constant(df["x"])
    glm_model = sm.OLS(df["y"], X).fit()
    print(glm_model.summary())
    X.to_csv("glm_X.csv", index=False)
    print("\n---\n")
