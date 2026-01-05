import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Synchronize Data
fg_df = pd.read_csv('fear_greed_index.csv')
hist_df = pd.read_csv('historical_data.csv')

fg_df['date'] = pd.to_datetime(fg_df['date'])
hist_df['trade_date'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
hist_df['trade_date'] = pd.to_datetime(hist_df['trade_date'])

df = pd.merge(hist_df, fg_df, left_on='trade_date', right_on='date', how='inner')
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

# 2. Pattern: Risk vs Sentiment (Volatility)
risk_analysis = df.groupby('classification')['Closed PnL'].std().reindex(sentiment_order)

# 3. Pattern: Top Account "Specialization"
closing = df[df['Closed PnL'] != 0].copy()
top_accounts = closing.groupby('Account')['Closed PnL'].sum().nlargest(10).index
account_bias = closing[closing['Account'].isin(top_accounts)].groupby(['Account', 'classification'])['Closed PnL'].mean().unstack().fillna(0)

# 4. Pattern: Taker vs Maker Performance
maker_taker = df.groupby(['classification', 'Crossed'])['Closed PnL'].mean().unstack()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Chart 1: Volatility (Risk)
sns.barplot(x=risk_analysis.index, y=risk_analysis.values, ax=axes[0,0], palette='Reds')
axes[0,0].set_title('PnL Volatility (Risk) by Sentiment')
axes[0,0].set_ylabel('Standard Deviation of PnL')

# Chart 2: Taker vs Maker Performance
maker_taker.loc[sentiment_order].plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Avg PnL: Taker (True) vs Maker (False)')

# Chart 3: Account Bias Heatmap
sns.heatmap(account_bias, annot=True, cmap='RdYlGn', fmt=".0f", ax=axes[1,0])
axes[1,0].set_title('Top 10 Accounts: Avg PnL by Sentiment')

# Chart 4: Trade Size Correlation
sns.scatterplot(data=closing[closing['Size USD'] < 100000], x='value', y='Closed PnL', alpha=0.1, ax=axes[1,1])
axes[1,1].set_title('Fear/Greed Value vs. PnL (Density)')

plt.tight_layout()
plt.savefig('advanced_trading_insights.png')
df.to_csv('final_trader_sentiment_analysis.csv', index=False)