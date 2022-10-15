"""
Helper Functions
"""

from typing import Dict, Tuple
import pandas as pd
import pandas.io.formats.style as style
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def is_close(num_1: float, num_2: float, rel_tol=1e-5, abs_tol=0.0) -> bool:
    """Compare two float numbers

    Args:
        rel_tol (float, optional): _description_. Defaults to 1e-5.
        abs_tol (float, optional): _description_. Defaults to 0.0.

    Returns:
        bool: True if two numbers are considered to be equal.
    """
    return abs(num_1 - num_2) <= max(rel_tol * max(abs(num_1), abs(num_2)), abs_tol)


def extract_sample_number(filename: str) -> str:
    """
    >>> extract_sample_number('cycle4_day11_2_3.csv)
    '2_3'
    >>> extract_sample_number('cycle4_day11_14_2.csv)
    '14_2
    """
    filename_token = filename.split("_")
    res = f"{filename_token[2]}_{filename_token[3]}"
    return res


def df_dict_generation_test(_dict: dict, repeat: int = 2):
    """Test function to generate dictionary including DataFrames.

    Args:
        _dict (dict): Python Dictionary: {"filename": pd.DataFrame}
        repeat (int, optional): Repeat number. Defaults to 2.
    """
    cnt = 0
    for key, value in _dict.items():
        print("=========================")
        print(f"File name: {key}")
        print("<DataFrame>")
        print(value.head())
        cnt += 1
        if cnt == repeat:
            break


def _find_deform(
    target_df: pd.DataFrame, bp_rp: pd.DataFrame, rel_tol=1e-5
) -> Tuple[pd.Series, pd.Series, np.int64, np.int64]:
    bioyield_point = bp_rp["Bioyield Point"]
    rupture_point = bp_rp["Rupture Point"]

    for force in target_df["Force"]:
        if is_close(force, bioyield_point.to_numpy(), rel_tol=rel_tol):
            idx_bp = target_df[target_df["Force"] == force].index.values[0]
            bp_deform = target_df["Stroke"][idx_bp]

    for force in target_df["Force"]:
        if is_close(force, rupture_point.to_numpy(), rel_tol=rel_tol):
            idx_rp = target_df[target_df["Force"] == force].index.values[0]
            rp_deform = target_df["Stroke"][idx_rp]

    return bp_deform, rp_deform, idx_bp, idx_rp


def plot_bp_and_rp(
    sample: str, utm_df_dict: Dict[str, pd.DataFrame], bp_rp_df: pd.DataFrame
) -> go.Figure:
    """Plot UTM Data and Points of BioYield and BioRupture

    Args:
        sample (str): Choose sample name
        utm_df_dict (Dict[str, pd.DataFrame]): UTM Data
        bp_rp_df (pd.DataFrame): Bioyield and Biorupture Point Data

    Returns:
        go.Figure: Plotly Figure
    """
    target_df = utm_df_dict[sample]
    bp_rp = bp_rp_df.loc[bp_rp_df["Name"] == sample]
    # bp = bp_rp["Bioyield Point"]
    # rp = bp_rp["Rupture Point"]

    # for force in target_df["Force"]:
    #     if is_close(force, bp.to_numpy()):
    #         idx_bp = target_df[target_df["Force"] == force].index.values[0]
    #         bp_deform = target_df["Stroke"][idx_bp]

    # for force in target_df["Force"]:
    #     if is_close(force, rp.to_numpy()):
    #         idx_rp = target_df[target_df["Force"] == force].index.values[0]
    #         rp_deform = target_df["Stroke"][idx_rp]

    bp_deform, rp_deform, idx_bp, idx_rp = _find_deform(target_df, bp_rp)

    df_bp = pd.DataFrame(
        dict(Deformation=bp_deform, Force=target_df["Force"][idx_bp]), index=[0]
    )
    df_rp = pd.DataFrame(
        dict(Deformation=rp_deform, Force=target_df["Force"][idx_rp]), index=[0]
    )

    df_new = pd.DataFrame(
        dict(Deformation=target_df["Stroke"], Force=target_df["Force"])
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_new["Deformation"], y=df_new["Force"], name="Main"))
    fig.add_trace(go.Scatter(x=df_bp["Deformation"], y=df_bp["Force"], name="Bioyield"))
    fig.add_trace(
        go.Scatter(x=df_rp["Deformation"], y=df_rp["Force"], name="Biorupture")
    )
    fig.update_xaxes(title="Deformation[mm]")
    fig.update_yaxes(title="Force[N")
    return fig


def plot_correlation(
    properties_df: pd.DataFrame,
    bp_rp_loc_df: pd.DataFrame,
    target: Tuple[str, str] = ("Weight", "Bioyield Point"),
    add_regression: bool = True,
    print_regression: bool = True,
) -> Tuple[float, float, np.float64, np.float64]:
    """Function to plot Scatter & Linear Regression

    Args:
        properties_df (pd.DataFrame): X
        bp_rp_loc_df (pd.DataFrame): Y
        target (Tuple[str, str], optional): Select X & Y. Defaults to ("Weight", "Bioyield Point").
        add_regression (bool, optional): True if you want to add regression. Defaults to True.
        print_regression (bool, optional): True it you want to print regression. Defaults to True.

    Returns:
        Tuple[float, float, np.float64, np.float64]: Regression outputs.
    """
    bp_rp_loc_df.fillna(bp_rp_loc_df.mean(), inplace=True)
    if add_regression:
        plot_df = pd.concat([properties_df, bp_rp_loc_df], axis=1)
        x_property = plot_df[target[0]].values.reshape(-1, 1)
        y_mechanical = plot_df[target[1]].values.reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(x_property, y_mechanical)
        y_hat = regr.predict(x_property)

        mse = mean_squared_error(y_mechanical, y_hat)
        _r2_score = r2_score(y_mechanical, y_hat)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=properties_df[target[0]],
            y=bp_rp_loc_df[target[1]],
            mode="markers",
            name="main",
        ),
        secondary_y=False,
    )
    if add_regression:
        fig.add_trace(
            go.Scatter(
                x=x_property.flatten(),
                y=y_hat.flatten(),
                mode="lines",
                name="LinearRegression",
            ),
            secondary_y=False,
        )
    fig.update_xaxes(title=f"{target[0]}[g]")
    fig.update_yaxes(title=f"{target[1]}[N]")
    if "Point" not in target[1]:
        fig.update_yaxes(title=f"{target[1]}")
    fig.show()
    if print_regression:
        print(f"y={regr.coef_.item():.3f}x+{regr.intercept_.item():.3f}")
        print(f"MSE : {mse}")
        print(f"R2 Score: {_r2_score}")
    return regr.coef_.item(), regr.intercept_.item(), mse, _r2_score


def secant_modulus(
    sample: str,
    utm_df_dict: Dict[str, pd.DataFrame],
    bp_rp_df: pd.DataFrame,
    rel_tol=1e-5,
) -> np.ndarray:
    """탄성계수를 구하기 위해 Secant Method를 사용한다.
    구하는 방법 : 원점과 Bioyield Point 사이의 기울기

    Args:
        sample (str): 구하고자 하는 Sample e.g. 4_3
        utm_df_dict (Dict[str, pd.DataFrame]): UTM 데이터
        bp_rp_df (pd.DataFrame): Properties 데이터

    Returns:
        np.ndarray: 기울기
    """
    target_df = utm_df_dict[sample]
    bp_rp = bp_rp_df.loc[bp_rp_df["Name"] == sample]
    bp_deform, _, idx_bp, _ = _find_deform(target_df, bp_rp, rel_tol=rel_tol)
    bp_force = target_df["Force"][idx_bp]  # np.float64
    secant = bp_force / bp_deform
    return secant


def evaluate_growth_level(properties_df: pd.DataFrame) -> pd.DataFrame:
    """Test function to simulate Growth Level Evaluation

    Args:
        properties_df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: returns dataframe with level added
    """

    def get_level(properties_df: pd.DataFrame) -> int:
        if properties_df["Weight"] < 100:
            level = 1
        elif properties_df["Weight"] < 150:
            level = 2
        elif properties_df["Weight"] < 200:
            level = 3
        else:
            level = 4
        return level

    properties_df["Growth_level"] = properties_df.apply(get_level, axis=1)
    return properties_df


def bp_rp_df_with_specific_loc(bp_rp_df: pd.DataFrame, loc: str = "2") -> pd.DataFrame:
    """Function to pick out specific location

    Args:
        bp_rp_df (pd.DataFrame): Dataframe of mechanical properties.
        loc (str, optional): 1=top, 2=middle, 3=bottom. Defaults to "2".

    Returns:
        pd.DataFrame: Dataframe of mechanical properties with location picked out.
    """
    bp_rp_loc_df = pd.DataFrame({"Bioyield Point": [], "Rupture Point": []})
    for i in bp_rp_df.index:
        if bp_rp_df.iloc[[i]]["Name"].values[0][-1] == loc:
            bp_rp_loc_df = pd.concat([bp_rp_loc_df, bp_rp_df.iloc[[i]]], axis=0)
    bp_rp_loc_df.reset_index(inplace=True)
    bp_rp_loc_df.drop(["index"], axis=1, inplace=True)
    return bp_rp_loc_df


def get_corr_matrix(
    df_target: pd.DataFrame,
    method: str = "pearson",
    draw_method: None | str = None,
    text_auto: bool = True,
) -> Tuple[pd.DataFrame, go.Figure | style.Styler]:
    """Get Correlation Matrix and Draw

    Args:
        df (pd.DataFrame): DataFrame to get Correlation Matrix
        method (None | str, optional): Correlation Calculation Method,
                                       {"pearson", "kendall", "spearman"}.
                                       Defaults to "pearson".
        draw_method (str, optional): Heatmap Draw Option. Defaults to "1".
        text_auto (bool, optional): Whether to display correlation values. Defaults to True.

    Returns:
        pd.DataFrame: Correlation Matrix
    """
    df_corr = df_target.corr(method=method)
    if draw_method:
        if draw_method == "1":
            styler = df_corr.style.background_gradient(cmap="coolwarm")
            ret = styler
        elif draw_method == "2":
            fig = px.imshow(df_corr, text_auto=text_auto)
            ret = fig

    return df_corr, ret
