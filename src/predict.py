import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller  # 添加这行导入

TAU = 0.0000062


# 3. 模拟分析
def simulation_analysis(df):
    # 创建新的预测结果列
    df["gamma_fixed"] = 0.0001  # 固定gamma值
    df["c_sim"] = df["LHS"] - (0.0001 * df["flow_launch"])
    df["LHS_pred"] = df["c_sim"] + (0.0001 * df["flow_launch"])

    # 创建发射过和未发射过的国家分组
    country_launches = df.groupby("country")["flow_launch"].sum()
    launched_countries = country_launches[country_launches > 0].index
    non_launched_countries = country_launches[country_launches == 0].index

    launched_data = df[df["country"].isin(launched_countries)]
    non_launched_data = df[df["country"].isin(non_launched_countries)]

    return df, launched_data, non_launched_data


def find_best_arima_order(series, max_p=3, max_d=2, max_q=3):
    """
    为时间序列找到最优的ARIMA模型阶数
    """
    # ADF检验
    adf_result = adfuller(series)
    d = 1 if adf_result[1] > 0.05 else 0

    # 尝试不同的p,d,q组合
    best_aic = float("inf")
    best_order_aic = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order_aic = (p, d, q)
            except:
                continue

    return best_order_aic


def forecast_analysis(launched_data):
    """
    修改后的预测分析函数
    """
    current_year = launched_data["year"].max()
    years_to_2050 = 2050 - current_year
    predictions = pd.DataFrame()
    arima_orders = {}  # 用于存储每个国家的ARIMA阶数

    for country in launched_data["country"].unique():
        country_data = launched_data[launched_data["country"] == country]

        try:
            c_series = country_data["c_sim"]

            if len(c_series) > 5:
                order = find_best_arima_order(c_series)
            else:
                order = (1, 0, 0)

            arima_orders[country] = order  # 记录ARIMA阶数

            model = ARIMA(c_series, order=order)
            results = model.fit()
            c_forecast = results.forecast(steps=years_to_2050)
            c_forecast = np.maximum(c_forecast, 1e-10)

            country_predictions = pd.DataFrame(
                {
                    "year": range(current_year + 1, 2051),
                    "country": country,
                    "c_pred": c_forecast,
                    "arima_order": str(order),
                }
            )

            last_lhs = country_data["LHS"].iloc[-1]
            country_predictions["flow_launch_pred"] = np.maximum(
                (last_lhs - country_predictions["c_pred"]) / 0.0001, 0
            )

            predictions = pd.concat([predictions, country_predictions])

        except Exception as e:
            print(f"预测{country}时出错: {str(e)}")
            continue

    # 在最后统一输出每个国家使用的ARIMA阶数
    print("\n各国使用的ARIMA模型阶数:")
    for country, order in arima_orders.items():
        print(f"{country}: ARIMA{order}")

    return predictions


# 5. 可视化函数
def plot_results(df, predictions):
    # 创建画布
    plt.figure(figsize=(15, 15))

    # c值预测趋势
    plt.subplot(3, 1, 1)
    for country in predictions["country"].unique():
        country_pred = predictions[predictions["country"] == country]
        plt.plot(country_pred["year"], country_pred["c_pred"], alpha=0.5, label=country)
    plt.title("Predicted c-values Trend (2023-2050)")
    plt.xlabel("Year")
    plt.ylabel("c-value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 发射量预测趋势
    plt.subplot(3, 1, 2)
    for country in predictions["country"].unique():
        country_pred = predictions[predictions["country"] == country]
        plt.plot(
            country_pred["year"],
            country_pred["flow_launch_pred"],
            alpha=0.5,
            label=country,
        )
    plt.title("Predicted Launch Flow Trend (2023-2050)")
    plt.xlabel("Year")
    plt.ylabel("Launch Flow")

    # c值在2035和2050年的分布
    plt.subplot(3, 1, 3)
    years_to_plot = [2035, 2050]
    c_data = [
        predictions[predictions["year"] == year]["c_pred"] for year in years_to_plot
    ]
    plt.boxplot(c_data, labels=["2035", "2050"])
    plt.title("Distribution of c-values in 2035 and 2050")
    plt.ylabel("c-value")

    plt.tight_layout()
    plt.show()


# 5. 可视化函数
def plot_results(df, predictions):
    # 创建画布
    plt.figure(figsize=(15, 15))

    # c值预测趋势
    plt.subplot(3, 1, 1)
    for country in predictions["country"].unique():
        country_pred = predictions[predictions["country"] == country]
        plt.plot(country_pred["year"], country_pred["c_pred"], alpha=0.5, label=country)
    plt.title("Predicted c-values Trend (2023-2050)")
    plt.xlabel("Year")
    plt.ylabel("c-value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 发射量预测趋势
    plt.subplot(3, 1, 2)
    for country in predictions["country"].unique():
        country_pred = predictions[predictions["country"] == country]
        plt.plot(
            country_pred["year"],
            country_pred["flow_launch_pred"],
            alpha=0.5,
            label=country,
        )
    plt.title("Predicted Launch Flow Trend (2023-2050)")
    plt.xlabel("Year")
    plt.ylabel("Launch Flow")

    # c值在2035和2050年的分布
    plt.subplot(3, 1, 3)
    years_to_plot = [2035, 2050]
    c_data = [
        predictions[predictions["year"] == year]["c_pred"] for year in years_to_plot
    ]
    plt.boxplot(c_data, labels=["2035", "2050"])
    plt.title("Distribution of c-values in 2035 and 2050")
    plt.ylabel("c-value")

    plt.tight_layout()
    plt.show()


def calculate_equilibrium_parameters(countries, tau=0.0000062, gamma=0.0001):
    """
    计算均衡所需的固定参数
    """
    # 计算每个国家的αi（由于gamma固定，所有国家的αi相同）
    alpha = 1 / (tau + gamma)
    alphas = {country: alpha for country in countries}

    # 计算分母项 1/τ + Σαj
    n = len(countries)
    denominator = (1 / tau) + (n * alpha)

    # 计算每个国家的λi（由于alpha相同，所有国家的λi也相同）
    lambda_i = alpha / denominator
    lambdas = {country: lambda_i for country in countries}

    print(f"\n参数计算过程:")
    print(f"tau = {tau}, gamma = {gamma}")
    print(f"alpha = 1/({tau} + {gamma}) = {alpha}")
    print(f"国家数量 n = {n}")
    print(f"分母 = 1/{tau} + {n} * {alpha} = {denominator}")
    print(f"lambda = {alpha} / {denominator} = {lambda_i}")

    return alphas, lambdas


def calculate_launches(predictions, alphas, lambdas):
    """
    根据预测的c值计算每个国家的发射量
    xi = αi[(1-ci) - Σλj(1-cj)]
    """
    new_predictions = predictions.copy()
    new_predictions["theoretical_flow_launch"] = 0.0
    new_predictions["sum_lambda_c"] = 0.0  # 添加新列

    for year in new_predictions["year"].unique():
        year_data = new_predictions[new_predictions["year"] == year]

        # 计算当年的Σλj(1-cj)
        sum_lambda_c = 0
        for _, row in year_data.iterrows():
            country = row["country"]
            c_j = row["c_pred"]
            term = lambdas[country] * (1 - c_j)
            sum_lambda_c += term

        # 将sum_lambda_c值添加到该年份的所有行
        new_predictions.loc[new_predictions["year"] == year, "sum_lambda_c"] = (
            sum_lambda_c
        )

        # 计算每个国家的发射量
        for country in year_data["country"].unique():
            country_mask = (new_predictions["year"] == year) & (
                new_predictions["country"] == country
            )
            c_i = new_predictions.loc[country_mask, "c_pred"].iloc[0]
            x_i = alphas[country] * ((1 - c_i) - sum_lambda_c)
            new_predictions.loc[country_mask, "theoretical_flow_launch"] = max(x_i, 0)

    return new_predictions


def simulation_analysis_major(df):
    """
    针对主要参与者的模拟分析函数
    """
    # 创建新的预测结果列
    df["gamma_fixed"] = 0.0001  # 固定gamma值
    df["c_sim"] = df["LHS"] - (0.0001 * df["flow_launch"])
    df["LHS_pred"] = df["c_sim"] + (0.0001 * df["flow_launch"])

    return df


def forecast_analysis_major(df):
    """
    针对主要参与者的预测分析函数
    """
    current_year = df["year"].max()
    years_to_2050 = 2050 - current_year
    predictions = pd.DataFrame()
    arima_orders = {}  # 用于存储每个国家的ARIMA阶数

    for country in df["country"].unique():
        country_data = df[df["country"] == country]

        try:
            c_series = country_data["c_sim"]

            if len(c_series) > 5:
                order = find_best_arima_order(c_series)
            else:
                order = (1, 0, 0)

            arima_orders[country] = order

            model = ARIMA(c_series, order=order)
            results = model.fit()
            c_forecast = results.forecast(steps=years_to_2050)
            c_forecast = np.maximum(c_forecast, 1e-10)

            country_predictions = pd.DataFrame(
                {
                    "year": range(current_year + 1, 2051),
                    "country": country,
                    "c_pred": c_forecast,
                    "arima_order": str(order),
                }
            )

            last_lhs = country_data["LHS"].iloc[-1]
            country_predictions["flow_launch_pred"] = np.maximum(
                (last_lhs - country_predictions["c_pred"]) / 0.0001, 0
            )

            predictions = pd.concat([predictions, country_predictions])

        except Exception as e:
            print(f"预测{country}时出错: {str(e)}")
            continue

    print("\n主要参与者的ARIMA模型阶数:")
    for country, order in arima_orders.items():
        print(f"{country}: ARIMA{order}")

    return predictions


def main_major_players():
    try:
        print("使用主要参与者数据进行分析")

        # 模拟分析
        df_sim = simulation_analysis_major(df_major)
        print("主要参与者模拟分析完成")

        # 预测分析
        predictions = forecast_analysis_major(df_sim)
        print("主要参与者预测分析完成")

        # 输出结果
        print("\n预测结果摘要:")
        print("\n2035年:")
        pred_2035 = predictions[predictions["year"] == 2035]
        print("\nc值统计:")
        print(pred_2035["c_pred"].describe())
        print("\n发射量统计:")
        print(pred_2035["flow_launch_pred"].describe())

        print("\n2050年:")
        pred_2050 = predictions[predictions["year"] == 2050]
        print("\nc值统计:")
        print(pred_2050["c_pred"].describe())
        print("\n发射量统计:")
        print(pred_2050["flow_launch_pred"].describe())

        # 可视化结果
        plot_results(df_sim, predictions)

        # 保存结果
        df_sim.to_csv("major_players_simulation_results.csv", index=False)
        predictions.to_csv("major_players_predictions_results.csv", index=False)

        print(
            "\n结果已保存到 'major_players_simulation_results.csv' 和 'major_players_predictions_results.csv'"
        )

        return df_sim, predictions

    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None, None
