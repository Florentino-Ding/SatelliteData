import os
import csv
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings
import concurrent.futures
import logging

warnings.filterwarnings('ignore')

# 初始化日志记录，将所有输出记录到日志文件中
logging.basicConfig(
    level=logging.INFO,
    filename="grid_search.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------
# Monte Carlo模拟函数（现增加tau和gamma参数）
def monte_carlo_simulation(df, tau, gamma, n_sims=100, error_range=0.1):
    all_results = []
    # 使用 tqdm 显示 Monte Carlo 模拟的进度条
    for sim in range(n_sims):
        sim_df = df.copy()
        mask = sim_df['flow_launch'] >= 1
        error_factors = np.random.uniform(1 - error_range, 1 + error_range, size=mask.sum())
        sim_df.loc[mask, 'X_bar'] = sim_df.loc[mask, 'flow_launch'] * error_factors
        sim_df.loc[~mask, 'X_bar'] = sim_df.loc[~mask, 'flow_launch']
        # 使用当前tau和gamma计算c_sim_bar
        sim_df['c_sim_bar'] = (1 - tau * sim_df['prev_total_launch'] -
                               tau * sim_df['annual_launch_total'] -
                               tau * sim_df['X_bar'] -
                               gamma * sim_df['X_bar'])
        sim_df['sim_id'] = sim
        all_results.append(sim_df)
    return pd.concat(all_results, ignore_index=True)

# ----------------------------
# 定义完整流程函数，输入tau和gamma，返回2024年预测结果的DataFrame
def run_pipeline(tau, gamma, n_sims_for_mc=50):
    # 1. 数据加载与预处理
    df = pd.read_csv("satellite_data.csv")
    df = df[df['year'] != 2025]  # 删除2025年的数据
    df = df.sort_values(['country', 'year'])
    df['prev_total_launch'] = df.groupby('country')['X_total_cum'].shift(1)
    df.loc[df['year'] == 1957, 'prev_total_launch'] = 0
    annual_total = df.groupby('year')['flow_launch'].sum().reset_index()
    annual_total.rename(columns={'flow_launch': 'annual_launch_total'}, inplace=True)
    annual_dict = dict(zip(annual_total['year'], annual_total['annual_launch_total']))
    df['annual_launch_total'] = df['year'].map(annual_dict)
    df['prev_total_launch'] = df['prev_total_launch'].fillna(0)
    
    # 计算边际成本 c_sim（依赖于tau和gamma）
    df['c_sim'] = 1 - tau * df['prev_total_launch'] - tau * df['annual_launch_total'] - (tau + gamma) * df['flow_launch']
    
    # 只保留有实际发射记录的国家
    country_launches = df.groupby('country')['flow_launch'].sum()
    launched_countries = country_launches[country_launches > 0].index.tolist()
    df_launch = df[df['country'].isin(launched_countries)].copy()
    
    # 2. Monte Carlo模拟计算c_sim的不确定性指标
    mc_results = monte_carlo_simulation(df_launch, tau, gamma, n_sims=n_sims_for_mc, error_range=0.1)
    c_sim_stats = mc_results.groupby(['country', 'year'])['c_sim_bar'].agg(
        lower_bound=lambda x: x.quantile(0.025),
        median=lambda x: x.median(),
        upper_bound=lambda x: x.quantile(0.975)
    ).reset_index()
    
    # 将蒙特卡洛统计结果合并进面板数据，并重命名相关列
    panel_data = df_launch.merge(c_sim_stats, on=['country', 'year'], how='left')
    panel_data = panel_data.rename(columns={
        'lower_bound': 'c_sim_lower',
        'median': 'c_sim_median',
        'upper_bound': 'c_sim_upper'
    })
    
    # 3. 筛选主要发射国（2014-2023至少5年有发射记录，且具有独立发射能力）
    temp_recent = panel_data[(panel_data['year'] >= 2014) & (panel_data['year'] <= 2023)]
    temp_capability = panel_data[(panel_data['year'] >= 2013) & (panel_data['year'] <= 2024)]
    capable_countries = temp_capability.groupby('country')['flow_launch'].max() >= 1
    capable_countries = capable_countries[capable_countries].index.tolist()
    country_launch_years = temp_recent.groupby('country')['flow_launch'].apply(lambda x: (x > 0).sum()).reset_index(name='launch_years')
    major_players = country_launch_years[
        (country_launch_years['launch_years'] > 5) & 
        (country_launch_years['country'].isin(capable_countries))
    ]['country'].tolist()
    major_player_data = panel_data[panel_data['country'].isin(major_players)].copy()
    major_player_data = major_player_data.sort_values(['country', 'year'])
    
    # 4. ARIMA模型预测——预测各国2024年的边际成本指标
    prediction_2024 = pd.DataFrame(columns=['country', 'c_pred_lower', 'c_pred_median', 'c_pred_upper'])
    fixed_order = (1, 0, 1)
    for country in major_player_data['country'].unique():
        country_data = major_player_data[major_player_data['country'] == country]
        if len(country_data[country_data['year'] <= 2023]) < 5:
            if not country_data[country_data['year'] == 2023].empty:
                last_year = country_data[country_data['year'] == 2023].iloc[0]
                new_row = {
                    'country': country,
                    'c_pred_lower': last_year['c_sim_lower'],
                    'c_pred_median': last_year['c_sim_median'],
                    'c_pred_upper': last_year['c_sim_upper']
                }
                prediction_2024 = pd.concat([prediction_2024, pd.DataFrame([new_row])], ignore_index=True)
            continue

        ts_data_lower = country_data[country_data['year'] <= 2023][['year', 'c_sim_lower']].set_index('year')
        ts_data_median = country_data[country_data['year'] <= 2023][['year', 'c_sim_median']].set_index('year')
        ts_data_upper = country_data[country_data['year'] <= 2023][['year', 'c_sim_upper']].set_index('year')
        try:
            forecast_lower = ARIMA(ts_data_lower['c_sim_lower'], order=fixed_order).fit().forecast(steps=1).iloc[0]
            forecast_median = ARIMA(ts_data_median['c_sim_median'], order=fixed_order).fit().forecast(steps=1).iloc[0]
            forecast_upper = ARIMA(ts_data_upper['c_sim_upper'], order=fixed_order).fit().forecast(steps=1).iloc[0]
            new_row = {
                'country': country,
                'c_pred_lower': forecast_lower,
                'c_pred_median': forecast_median,
                'c_pred_upper': forecast_upper
            }
            prediction_2024 = pd.concat([prediction_2024, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            if not country_data[country_data['year'] == 2023].empty:
                last_year = country_data[country_data['year'] == 2023].iloc[0]
                new_row = {
                    'country': country,
                    'c_pred_lower': last_year['c_sim_lower'],
                    'c_pred_median': last_year['c_sim_median'],
                    'c_pred_upper': last_year['c_sim_upper']
                }
                prediction_2024 = pd.concat([prediction_2024, pd.DataFrame([new_row])], ignore_index=True)

    # 检查预测结果中24年美国的c_sim_median
    us_prediction = prediction_2024[prediction_2024['country'] == 'United States']
    if not us_prediction.empty and us_prediction.iloc[0]['c_pred_median'] < 0.2:
        raise ValueError("Abandon candidate: 24年美国的c_sim_median 小于0.2")
    
    # 5. 利用预测的成本计算2024年的发射量预测
    pred_2024 = pd.DataFrame(columns=[
        'country', 
        'flow_launch_pred_lower', 
        'flow_launch_pred_median', 
        'flow_launch_pred_upper', 
        'flow_launch_actual'
    ])
    df_2023 = major_player_data[major_player_data['year'] == 2023]
    S_prev = df_2023['annual_launch_total'].sum() if not df_2023.empty else 0
    df_2024 = major_player_data[major_player_data['year'] == 2024]

    for scenario in ['lower', 'median', 'upper']:
        temp_data = pd.DataFrame(columns=['country', 'c_pred', 'd'])
        for country in prediction_2024['country'].unique():
            c_pred = prediction_2024[prediction_2024['country'] == country][f'c_pred_{scenario}'].iloc[0]
            d = 1 / (tau + gamma)
            temp_data = pd.concat([temp_data, pd.DataFrame([{'country': country, 'c_pred': c_pred, 'd': d}])],
                                  ignore_index=True)
        sum_d = temp_data['d'].sum()
        temp_data['lambda'] = temp_data['d'] / (1 / tau + sum_d)
        sum_lambda = temp_data['lambda'].sum()
        sum_lambda_c = (temp_data['lambda'] * temp_data['c_pred']).sum()
        for country in temp_data['country']:
            row = temp_data[temp_data['country'] == country].iloc[0]
            d = row['d']
            c_pred = row['c_pred']
            # 发射量预测公式
            flow_launch_pred = d * ((1 - tau * S_prev) * (1 - sum_lambda) - c_pred + sum_lambda_c)
            if country in pred_2024['country'].values:
                pred_2024.loc[pred_2024['country'] == country, f'flow_launch_pred_{scenario}'] = flow_launch_pred
            else:
                pred_2024 = pd.concat([pred_2024, pd.DataFrame([{'country': country, f'flow_launch_pred_{scenario}': flow_launch_pred}])],
                                      ignore_index=True)

    # 加入2024年的实际发射量数据（若存在）
    for country in df_2024['country'].unique():
        flow_launch_actual = df_2024[df_2024['country'] == country]['flow_launch'].iloc[0]
        if country in pred_2024['country'].values:
            pred_2024.loc[pred_2024['country'] == country, 'flow_launch_actual'] = flow_launch_actual
        else:
            pred_2024 = pd.concat([pred_2024, pd.DataFrame([{'country': country, 'flow_launch_actual': flow_launch_actual}])],
                                  ignore_index=True)
    
    return pred_2024

# ----------------------------
# 评价函数：计算预测区间覆盖率

def evaluate_coverage(pred_df):
    # 只考虑有实际发射量数据且大于1的样本
    valid = pred_df.dropna(subset=['flow_launch_actual'])
    valid = valid[valid['flow_launch_actual'] >= 1]
    
    if valid.empty:
        return 0
        
    coverage_count = 0
    total = 0
    
    for idx, row in valid.iterrows():
        if pd.isna(row['flow_launch_pred_lower']) or pd.isna(row['flow_launch_pred_upper']):
            continue
            
        total += 1
        # 注意：这里的判断条件应该是检查实际值是否在预测区间内
        # 原代码有逻辑错误，修正为下面的条件
        if row['flow_launch_pred_upper'] <= row['flow_launch_actual'] <= row['flow_launch_pred_lower']:
            coverage_count += 1
            
    return coverage_count / total if total > 0 else 0
# ----------------------------
# 定义评价候选参数组的函数，供多进程调用。同时计算覆盖率与预测中位数误差 (Mean Absolute Error, MAE)
def calculate_top10_mae(pred_df):
    valid = pred_df.dropna(subset=['flow_launch_actual', 'flow_launch_pred_median'])
    if valid.empty:
        return float('inf')
    top10_countries = valid.groupby('country')['flow_launch_actual'].max().nlargest(10).index
    top10_data = valid[valid['country'].isin(top10_countries)]
    return (top10_data['flow_launch_pred_median'] - top10_data['flow_launch_actual']).abs().mean()

def evaluate_candidate(args):
    tau, gamma, n_sims = args
    try:
        pred_df = run_pipeline(tau, gamma, n_sims_for_mc=n_sims)
        coverage = evaluate_coverage(pred_df)
        error = calculate_top10_mae(pred_df)
    except Exception as e:
        logger.error(f"Error evaluating candidate tau={tau:.16f}, gamma={gamma:.16f}: {e}")
        coverage = 0
        error = float('inf')
    return (tau, gamma, coverage, error)

# ----------------------------
# 网格搜索部分（增加检查点保存、中断恢复、多进程计算，并保存预测误差最小的15个候选组合）
if __name__ == "__main__":
    import sys

    checkpoint_file = 'grid_search_checkpoint.csv'
    
    # 加载已有的检查点文件，如果存在
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_set = set(checkpoint_df.apply(lambda row: f"{float(row['tau']):.16f}_{float(row['gamma']):.16f}", axis=1).tolist())
        if not checkpoint_df.empty:
            best_row = checkpoint_df.loc[checkpoint_df['coverage'].idxmax()]
            best_score = float(best_row['coverage'])
            best_tau = float(best_row['tau'])
            best_gamma = float(best_row['gamma'])
        else:
            best_score = -1
            best_tau = None
            best_gamma = None
    else:
        processed_set = set()
        best_score = -1
        best_tau = None
        best_gamma = None

    # 定义候选参数范围（可根据需要调整）
    tau_candidates = np.linspace(1e-08, 1e-09, 200)
    gamma_candidates = np.linspace(0.00025, 0.000075, 10)
    n_sims_for_mc = 100  # 加速搜索时使用较少的模拟次数

    # 构造所有候选组合列表，跳过已处理的组合
    candidates = [(tau, gamma, n_sims_for_mc) for tau, gamma in itertools.product(tau_candidates, gamma_candidates)
                  if f"{tau:.16f}_{gamma:.16f}" not in processed_set]

    logger.info("开始多进程网格搜索最优参数。。。")
    
    # 打开检查点文件（追加模式），写入表头（如果文件不存在）
    if not os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tau', 'gamma', 'coverage', 'error'])
    
    checkpoint_f = open(checkpoint_file, 'a', newline='')
    checkpoint_writer = csv.writer(checkpoint_f)
    
    # 并行计算候选参数结果，处理由于美国c_sim_median小于0.2引起的ValueError
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_candidate = {executor.submit(evaluate_candidate, candidate): candidate for candidate in candidates}
            # 用 tqdm 包装 as_completed 的生成器，显示候选组合的进度
            for future in tqdm(concurrent.futures.as_completed(future_to_candidate),
                               total=len(future_to_candidate),
                               desc="Evaluating Candidates"):
                try:
                    tau_val, gamma_val, coverage, error = future.result()
                except ValueError as ve:
                    logger.error(f"Candidate abandoned due to US c_sim_median error: {ve}")
                    continue
                key = f"{tau_val:.16f}_{gamma_val:.16f}"
                if coverage > best_score:
                    best_score = coverage
                    best_tau = tau_val
                    best_gamma = gamma_val
                # 写入当前候选组合结果并立即刷新
                checkpoint_writer.writerow([f"{tau_val:.16f}", f"{gamma_val:.16f}", coverage, error])
                checkpoint_f.flush()
                logger.info(f"候选 tau={tau_val:.16f}, gamma={gamma_val:.16f} -> 覆盖率: {coverage:.2%}, 误差: {error:.4f}")
    except KeyboardInterrupt:
        logger.info("用户中断了搜索，检查点已保存，下次可恢复运行。")
    finally:
        checkpoint_f.close()

    logger.info(f"\n最佳参数（基于覆盖率）为：")
    logger.info(f"tau = {best_tau:.16f}, gamma = {best_gamma:.16f}，覆盖率 = {best_score:.2%}")
    
    # 保存预测误差最小的15个候选组合，同时保存其覆盖率与误差
    checkpoint_df = pd.read_csv(checkpoint_file)
    checkpoint_df = checkpoint_df[checkpoint_df['error'] != float('inf')]
    top10 = checkpoint_df.sort_values(by="error", ascending=True).head(10)
    top10.to_csv("top10_candidates.csv", index=False)
    logger.info("保存预测结果和实际结果差异最小的10个候选组合至文件：top10_candidates.csv")
    
    # 使用最优参数运行完整流程（增加蒙特卡洛模拟次数以获得更稳健的预测）
    try:
        final_pred_2024 = run_pipeline(best_tau, best_gamma, n_sims_for_mc=500)
        final_coverage = evaluate_coverage(final_pred_2024)
        final_pred_2024.to_csv('flow_launch_predictions_2024_optimal.csv', index=False)
        logger.info(f"最终使用最优参数时的覆盖率: {final_coverage:.2%}")
        logger.info("最终预测结果已保存到文件：flow_launch_predictions_2024_optimal.csv")
    except ValueError as ve:
        logger.error(f"Candidate abandoned due to US c_sim_median error: {ve}")
        logger.info("未找到覆盖率非零的组合。")