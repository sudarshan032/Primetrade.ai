import pandas as pd
import numpy as np
from scipy.stats import norm
import ast
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('TRADES_CopyTr_90D_ROI.csv')

df['Trade_History'] = df['Trade_History'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def extract_trade_data(trade_history, port_id):
    trades = []
    for trade in trade_history:
        trade_data = trade.copy()
        trade_data['Port_IDs'] = port_id
        trade_data['Position_Type'] = f"{trade['side']}_{trade['positionSide']}"
        trades.append(trade_data)
    return trades

all_trades = []
for _, row in df.iterrows():
    if isinstance(row['Trade_History'], list):
        all_trades.extend(extract_trade_data(row['Trade_History'], row['Port_IDs']))

df_trades = pd.DataFrame(all_trades)

df_trades['PnL'] = df_trades['realizedProfit']
df_trades['Investment'] = df_trades['quantity'] * df_trades['price']
df_trades['ROI'] = (df_trades['PnL'] / df_trades['Investment']) * 100

sharpe_ratios = {}
for port_id, group in df_trades.groupby('Port_IDs'):
    mean_returns = group['PnL'].mean()
    std_dev_returns = group['PnL'].std()
    sharpe_ratios[port_id] = (mean_returns / std_dev_returns) if std_dev_returns != 0 else np.nan

df_trades['cumulative_profit'] = df_trades.groupby('Port_IDs')['PnL'].cumsum()
df_trades['peak'] = df_trades.groupby('Port_IDs')['cumulative_profit'].cummax()
df_trades['drawdown'] = (df_trades['cumulative_profit'] - df_trades['peak']) / df_trades['peak']
max_drawdowns = df_trades.groupby('Port_IDs')['drawdown'].min()

df_trades['Win'] = df_trades['PnL'] > 0
win_positions = df_trades.groupby('Port_IDs')['Win'].sum()
total_positions = df_trades.groupby('Port_IDs').size()
win_rates = (win_positions / total_positions).round(2)

metrics = pd.DataFrame({
    'Port_IDs': win_positions.index,
    'Total_PnL': df_trades.groupby('Port_IDs')['PnL'].sum().round(3),
    'Total_ROI': df_trades.groupby('Port_IDs')['ROI'].mean(),
    'Sharpe_Ratio': pd.Series(sharpe_ratios),
    'Max_Drawdown': max_drawdowns,
    'Win_Rate': win_rates,
    'Total_Positions': total_positions,
    'Win_Positions': win_positions
}).reset_index(drop=True)

weights = {
    'Total_ROI': 0.3,
    'Total_PnL': 0.25,
    'Sharpe_Ratio': 0.2,
    'Win_Rate': 0.15,
    'Max_Drawdown': -0.1 
}

for metric, weight in weights.items():
    metrics[metric + '_Score'] = metrics[metric] * weight

metrics['Rank_Score'] = metrics[[metric + '_Score' for metric in weights]].sum(axis=1)

metrics = metrics.sort_values(by='Rank_Score', ascending=False)

output_file = 'final_account_ranking.csv'
metrics.to_csv(output_file, index=False)

plt.figure(figsize=(18, 10))
sns.barplot(x=metrics['Port_IDs'].head(20), y=metrics['Rank_Score'].head(20))
plt.title('Top 20 Accounts by Rank Score')
plt.xlabel('Port_IDs')
plt.ylabel('Rank Score')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
plt.savefig('top_20_accounts.png')

top_20_accounts = metrics.head(20)
print(top_20_accounts)
