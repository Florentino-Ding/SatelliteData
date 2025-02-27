import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.predict import *

# 1. 设置工作目录和导入数据
DATA_PATH = ".cache/satellite_data.csv"
# 重新定义 tau
TAU = 0.0000062

# 使用pandas读取Excel文件
df = pd.read_csv(DATA_PATH)

# 删除2025年的数据
df = df[df["year"] != 2025]
print(f"删除2025年数据后的数据集大小: {df.shape}")
print("\n年份范围：")
print(df["year"].unique())


# 重新基于一阶条件计算左侧变量
df["LHS"] = 1 - TAU * (df["X_total_cum"] + df["flow_launch"])

# 假设 df 是你的 DataFrame
df = df.sort_values(["country", "year"])

# 创建存储估计结果的列
df["c_hat"] = float("nan")
df["gamma_hat"] = float("nan")
df["c_hat_constrained"] = float("nan")
df["gamma_hat_constrained"] = float("nan")

# 获取所有国家的列表
countries = df["country"].unique()

# 按国家逐步估计参数
for country in countries:
    # 获取当前国家的数据
    country_data = df[df["country"] == country]

    # 检查是否有足够的观测值（至少2个）
    if len(country_data[country_data["stock_launch"] != 0]) >= 2:
        # 进行回归
        X = sm.add_constant(country_data["stock_launch"])
        y = country_data["LHS"]
        try:
            model = sm.OLS(y, X).fit()
            # 将结果存储回DataFrame
            df.loc[df["country"] == country, "c_hat"] = model.params["const"]
            df.loc[df["country"] == country, "gamma_hat"] = model.params["stock_launch"]
        except:
            continue

# 约束估计
for country in countries:
    country_data = df[df["country"] == country]

    # 检查是否有足够的观测值
    if (
        country_data["c_hat"].notna().sum() > 0
        and country_data["gamma_hat"].notna().sum() > 0
    ):
        # OLS 估计 c_hat
        try:
            X = sm.add_constant(country_data["stock_launch"])
            y = country_data["c_hat"]
            model = sm.OLS(y, X).fit()
            c_hat_constrained = (
                model.params["const"]
                + model.params["stock_launch"] * country_data["stock_launch"]
            )
            df.loc[df["country"] == country, "c_hat_constrained"] = (
                c_hat_constrained.clip(lower=0)
            )
        except:
            continue

        # OLS 估计 gamma_hat
        try:
            y = country_data["gamma_hat"]
            model = sm.OLS(y, X).fit()
            gamma_hat_constrained = (
                model.params["const"]
                + model.params["stock_launch"] * country_data["stock_launch"]
            )
            df.loc[df["country"] == country, "gamma_hat_constrained"] = (
                gamma_hat_constrained.clip(lower=0.01)
            )
        except:
            continue


# 保存结果到 CSV 文件
df.to_csv("parameter_estimation_results.csv", index=False)
print("结果已保存到 'parameter_estimation_results.csv'")


# 主函数
def main():
    try:
        print("Using existing processed data")

        # 3. 模拟分析
        df_sim, launched_data, non_launched_data = simulation_analysis(df)
        print("Simulation analysis completed")

        # 4. 预测分析
        predictions = forecast_analysis(launched_data)
        print("Forecast analysis completed")

        # 5. 输出结果
        print("\nPrediction Summary:")
        print("\nYear 2035:")
        pred_2035 = predictions[predictions["year"] == 2035]
        print("\nc-value statistics:")
        print(pred_2035["c_pred"].describe())
        print("\nLaunch flow statistics:")
        print(pred_2035["flow_launch_pred"].describe())

        print("\nYear 2050:")
        pred_2050 = predictions[predictions["year"] == 2050]
        print("\nc-value statistics:")
        print(pred_2050["c_pred"].describe())
        print("\nLaunch flow statistics:")
        print(pred_2050["flow_launch_pred"].describe())

        # 6. 可视化结果
        plot_results(df_sim, predictions)

        # 7. 保存结果到CSV文件
        df_sim.to_csv("simulation_results.csv", index=False)
        predictions.to_csv("predictions_results.csv", index=False)

        # 8. 输出详细数据
        print("\n模拟结果数据预览（前5行）：")
        print(df_sim.head())
        print("\n预测结果数据预览（前5行）：")
        print(predictions.head())

        # 9. 输出基本统计信息
        print("\n模拟结果基本统计：")
        print(df_sim.describe())
        print("\n预测结果基本统计：")
        print(predictions.describe())

        return df_sim, predictions

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None


# 运行分析a
df_result, predictions = main()

# 主要计算过程
countries = predictions["country"].unique()
print(f"\n总计{len(countries)}个国家参与计算")

# 计算均衡参数
alphas, lambdas = calculate_equilibrium_parameters(countries)

# 保存alpha和lambda值
alpha_lambda_df = pd.DataFrame(
    {
        "country": list(alphas.keys()),
        "alpha": list(alphas.values()),
        "lambda": list(lambdas.values()),
    }
)
alpha_lambda_df.to_csv("alpha_lambda_values.csv", index=False)
print("\nAlpha和Lambda值已保存到 'alpha_lambda_values.csv'")

# 计算发射量
new_predictions = calculate_launches(predictions, alphas, lambdas)

# 输出结果统计
print("\n基于理论公式的发射量预测统计:")
print("\n2035年:")
pred_2035 = new_predictions[new_predictions["year"] == 2035]
print(pred_2035[["country", "c_pred", "theoretical_flow_launch"]].to_string())

print("\n2050年:")
pred_2050 = new_predictions[new_predictions["year"] == 2050]
print(pred_2050[["country", "c_pred", "theoretical_flow_launch"]].to_string())

# 保存新的预测结果
new_predictions.to_csv("theoretical_predictions_results.csv", index=False)
print("\n新的预测结果已保存到 'theoretical_predictions_results.csv'")


# 导入基础数据
df = pd.read_csv("simulation_results.csv")

# 删除指定的列
columns_to_drop = [
    "c_hat",
    "gamma_hat",
    "gamma_hat_constrained",
    "c_hat_constrained",
    "LHS_pred",
]
df = df.drop(columns=columns_to_drop)

# 计算每年的cost_global（基于flow_launch=0的样本）
cost_global_by_year = df[df["flow_launch"] == 0].groupby("year")["c_sim"].first()

# 为每个样本添加对应年份的cost_global
df["cost_global"] = df["year"].map(cost_global_by_year)

# 生成launch_status变量
df["launch_status"] = (df["flow_launch"] > 0).astype(int)

# 保存新的数据集
df.to_csv("simulation_latestlaunch.csv", index=False)

# 输出处理结果摘要
print("数据处理完成：")
print(f"总样本数: {len(df)}")
print("\n前5行数据预览：")
print(df.head())
print("\n每年的cost_global值：")
print(cost_global_by_year)
print("\nlaunch_status的分布：")
print(df["launch_status"].value_counts())


# Calculate launch frequency between 2015-2024
launch_freq = (
    df[(df["year"] >= 2015) & (df["year"] <= 2024)]
    .groupby("country")["launch_status"]
    .sum()
    .reset_index()
)
launch_freq.columns = ["country", "frequency_launch"]

# Merge frequency info with 2024 data
df_2024 = df[df["year"] == 2024].copy()
df_2024 = df_2024.merge(launch_freq, on="country", how="left")

# Create figure
plt.figure(figsize=(15, 6))

# Subplot 1: Distribution for all countries
plt.subplot(1, 2, 1)
plt.hist(df_2024["frequency_launch"], bins=20, edgecolor="black")
plt.title("Launch Frequency Distribution - All Countries (2015-2024)")
plt.xlabel("Launch Frequency")
plt.ylabel("Number of Countries")

# Add descriptive statistics
stats_all = df_2024["frequency_launch"].describe()
plt.text(
    0.7,
    0.95,
    f'Mean: {stats_all["mean"]:.2f}\nStd: {stats_all["std"]:.2f}\n'
    + f'Min: {stats_all["min"]:.0f}\nMax: {stats_all["max"]:.0f}\n'
    + f"Total Countries: {len(df_2024)}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

# Subplot 2: Distribution for active countries only
df_2024_active = df_2024[df_2024["frequency_launch"] > 0]
plt.subplot(1, 2, 2)
plt.hist(df_2024_active["frequency_launch"], bins=20, edgecolor="black")
plt.title("Launch Frequency Distribution - Active Countries (2015-2024)")
plt.xlabel("Launch Frequency")
plt.ylabel("Number of Countries")

# Add descriptive statistics
stats_active = df_2024_active["frequency_launch"].describe()
plt.text(
    0.7,
    0.95,
    f'Mean: {stats_active["mean"]:.2f}\nStd: {stats_active["std"]:.2f}\n'
    + f'Min: {stats_active["min"]:.0f}\nMax: {stats_active["max"]:.0f}\n'
    + f"Total Countries: {len(df_2024_active)}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nLaunch Frequency Statistics - All Countries:")
print(stats_all)
print("\nLaunch Frequency Statistics - Active Countries:")
print(stats_active)

# Save frequency data
launch_freq.to_csv("launch_frequency_2015_2024.csv", index=False)
print("\nLaunch frequency data saved to 'launch_frequency_2015_2024.csv'")

# Print top 10 most active countries
print("\nTop 10 Countries by Launch Frequency:")
print(launch_freq.nlargest(10, "frequency_launch"))


# 筛选发射频率大于等于5的国家
major_players = launch_freq[launch_freq["frequency_launch"] >= 5]

# 打印主要参与者信息
print("Major Players (Launch Frequency >= 5):")
print(f"Total number of major players: {len(major_players)}")
print("\nDetailed list of major players:")
print(major_players.sort_values("frequency_launch", ascending=False))

# 使用这些主要参与者筛选原始数据
major_players_list = major_players["country"].tolist()
df_major = df[df["country"].isin(major_players_list)].copy()

# 打印数据集信息
print(f"\nOriginal dataset size: {len(df)}")
print(f"Major players dataset size: {len(df_major)}")
print(f"Percentage of data retained: {(len(df_major)/len(df)*100):.2f}%")

# 保存主要参与者名单
major_players.to_csv("major_players_list.csv", index=False)
print("\nMajor players list saved to 'major_players_list.csv'")

# 保存主要参与者的完整数据
df_major.to_csv("major_players_data.csv", index=False)
print("Major players complete data saved to 'major_players_data.csv'")

# 显示每个主要参与者的发射频率分布
plt.figure(figsize=(12, 6))
plt.bar(range(len(major_players)), major_players["frequency_launch"])
plt.xticks(range(len(major_players)), major_players["country"], rotation=45, ha="right")
plt.title("Launch Frequency Distribution Among Major Players (2015-2024)")
plt.xlabel("Country")
plt.ylabel("Launch Frequency")
plt.tight_layout()
plt.show()


# 读取major players数据
df_major = pd.read_csv("major_players_data.csv")


# 运行主要参与者分析
df_major_result, major_predictions = main_major_players()


# 使用major_players的预测结果进行均衡计算

# 读取预测数据
major_predictions = pd.read_csv("major_players_predictions_results.csv")

# 计算主要参与者的均衡参数
major_countries = major_predictions["country"].unique()
print(f"\n总计{len(major_countries)}个主要参与国家参与计算")

# 计算均衡参数 (使用相同的tau和gamma值)
major_alphas, major_lambdas = calculate_equilibrium_parameters(
    major_countries, tau=0.0000062, gamma=0.0001
)

# 保存主要参与者的alpha和lambda值
major_alpha_lambda_df = pd.DataFrame(
    {
        "country": list(major_alphas.keys()),
        "alpha": list(major_alphas.values()),
        "lambda": list(major_lambdas.values()),
    }
)
major_alpha_lambda_df.to_csv("major_players_alpha_lambda_values.csv", index=False)
print("\n主要参与者的Alpha和Lambda值已保存到 'major_players_alpha_lambda_values.csv'")

# 计算主要参与者的理论发射量
major_theoretical_predictions = calculate_launches(
    major_predictions, major_alphas, major_lambdas
)

# 输出主要参与者的结果统计
print("\n主要参与者基于理论公式的发射量预测统计:")
print("\n2035年:")
major_pred_2035 = major_theoretical_predictions[
    major_theoretical_predictions["year"] == 2035
]
print(major_pred_2035[["country", "c_pred", "theoretical_flow_launch"]].to_string())

print("\n2050年:")
major_pred_2050 = major_theoretical_predictions[
    major_theoretical_predictions["year"] == 2050
]
print(major_pred_2050[["country", "c_pred", "theoretical_flow_launch"]].to_string())

# 保存主要参与者的新预测结果
major_theoretical_predictions.to_csv(
    "major_players_theoretical_predictions_results.csv", index=False
)
print(
    "\n主要参与者的新预测结果已保存到 'major_players_theoretical_predictions_results.csv'"
)

# 添加结果比较
print("\n预测结果比较分析:")
for year in [2035, 2050]:
    year_data = major_theoretical_predictions[
        major_theoretical_predictions["year"] == year
    ]
    print(f"\n{year}年预测统计:")
    print("\n原始预测发射量统计:")
    print(year_data["flow_launch_pred"].describe())
    print("\n理论模型发射量统计:")
    print(year_data["theoretical_flow_launch"].describe())
