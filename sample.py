import numpy as np
import pandas as pd


def calculate_price_with_tax_rate(price: float, tax_rate: float) -> float:
    """税込計算をする関数"""
    return price * (1 + tax_rate)


def fill_na_with_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """指定した列の欠損値を最頻値で埋める関数"""
    mode_value = df[column].mode()[0]
    return df[column].fillna(mode_value)
