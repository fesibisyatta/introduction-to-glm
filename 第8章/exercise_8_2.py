import pandas as pd
import numpy as np
import itertools

from mlogit import MultinomialLogitOptimizer

if __name__ == "__main__":

    index = [
    ("Low", "Low"),
    ("Low", "High"),
    ("Medium", "Low"),
    ("Medium", "High"),
    ("High", "Low"),
    ("High", "High")
    ]

    mi = pd.MultiIndex.from_tuples(index, names=["Satisfaction", "Contact"])
    df_satisfaction = pd.DataFrame(
        [
            [65, 34, 54, 47, 100, 100],
            [130, 141, 76, 116, 111, 191],
            [67, 130, 48, 105, 62, 104]
        ],
        columns = mi,
        index = ["Tower_block", "Apartment", "House"],
    )
    
    # MultiIndex列を縦持ちに変換
    df_long = (
        df_satisfaction
        .stack(level=["Satisfaction", "Contact"])
        .reset_index()
        .rename(columns={
            "level_0": "House",
            0: "num_people"
        })
    )

    results = {}
    configs = {
        "X1": {"cat_cols": ["House"], "drop": ["House_Apartment", "Satisfaction_Low"], "interaction": False},
        "X2": {"cat_cols": ["Contact"], "drop": ["Contact_Low", "Satisfaction_Low"], "interaction": False},
        "X3": {"cat_cols": ["House", "Contact"], "drop": ["House_Apartment", "Contact_Low", "Satisfaction_Low"], "interaction": False},
        "X4": {"cat_cols": ["House", "Contact"], "drop": ["House_Apartment", "Contact_Low", "Satisfaction_Low"], "interaction": True}
    }

    for key, cfg in configs.items():

        X = pd.get_dummies(
            df_long[cfg["cat_cols"] + ["Satisfaction", "num_people"]], dtype=int, columns=cfg["cat_cols"] + ["Satisfaction"]
        ).drop(cfg["drop"], axis=1)

        # 交互作用（X4 のみ）
        if cfg["interaction"]:
            X["House_House:Contact_High"] = X["House_House"] & X["Contact_High"]
            X["House_Tower_block:Contact_High"] = X["House_Tower_block"] & X["Contact_High"]
            
        #集計
        list_expvar = [v for v in X.columns if (v != "num_people") & ("Satisfaction" not in v)]
        a = X.groupby(list_expvar, as_index = False).sum()[list_expvar+ ["num_people"]]
        b = X[X["Satisfaction_High"] == 1].groupby(list_expvar, as_index = False).sum()[list_expvar + ["num_people"]].rename(columns = {"num_people":"Satisfaction_High"})
        c = X[X["Satisfaction_Medium"] == 1].groupby(list_expvar, as_index = False).sum()[list_expvar + ["num_people"]].rename(columns = {"num_people":"Satisfaction_Medium"})
        df_tmp = pd.merge(pd.merge(a, b, on = list_expvar, how = "left"), c, on = list_expvar, how = "left")

        X = df_tmp[list_expvar].values
        n = df_tmp["num_people"].values
        y = df_tmp[["Satisfaction_Medium", "Satisfaction_High"]].values
        X = np.column_stack([np.ones(len(X)), X])
        results[key] = {
            "X": X, 
            "y":y,
            "n": n, 
            "coef": ["intercept"] + list_expvar,
            "class_label": ["Satisfaction_Medium","Satisfaction_High"]
        }
    
    # 各モデルの推定
    for model, res in results.items():
        
        cls = MultinomialLogitOptimizer(results[model]["X"], results[model]["y"], results[model]["n"])
        cls.fit()
        beta, std_error = cls.beta, cls.std_error
        columns = [f"{b}_{a}" for (a, b) in itertools.product(results[model]["class_label"], results[model]["coef"])]
        print(" ")
        print(" ")
        print(f"モデル {model} の結果")
        print(pd.DataFrame([beta.flatten() , std_error.flatten()],columns = columns, index = ["推定値", "標準誤差"]).T)
        print(" ")
        print("逸脱度：", cls.calc_deviance())
        print("pearsonのカイ二乗統計量：", cls.calc_chi2_statistic())

