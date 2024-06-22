#################################################
    # Dataset
#################################################
import pandas as pd
from utils.helper import eda

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# read the csv file
df = pd.read_csv("dataset/oil.csv")
df = df[6:]

df = df[['brent', 'WTI', 'SENT']]  # Price, WTI, SENT, GRACH
eda(df, 'brent')

# Initialize the scaler
scaler = MinMaxScaler()

# Scale the 'brent', 'WTI', and 'SENT' columns
df[['brent', 'WTI', 'SENT']] = scaler.fit_transform(df[['brent', 'WTI', 'SENT']])



# Plotting the scaled columns
plt.figure(figsize=(10, 6))
plt.plot( df['brent'], label='Brent')
plt.plot( df['WTI'], label='WTI')
plt.plot( df['SENT'], label='Sentiment')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Scaled Values')
plt.title('Brent, WTI, and Sentiment Scores')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Assuming df is your DataFrame
brent = df['brent']
sent = df['SENT']

# Calculate the line of best fit
slope, intercept, r_value, p_value, std_err = linregress(sent, brent)
line = slope * sent + intercept

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(sent, brent, label='Data points', color='blue', alpha=0.5)
plt.plot(sent, line, label=f'Best fit line (r={r_value:.2f})', color='red')

# Formatting the plot
plt.xlabel('Sentiment Scores')
plt.ylabel('Brent Prices')
plt.title('Correlation between Brent Prices and Sentiment Scores')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Create the scatter plot with a regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=sent, y=brent, ci=None, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})

# Formatting the plot
plt.xlabel('Sentiment Scores')
plt.ylabel('Brent Prices')
plt.title('Correlation between Brent Prices and Sentiment Scores')
plt.grid(True)

# Show the plot

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sent, y=brent, alpha=0.5, label='Data points', color='blue')

# Calculate the Spearman correlation
spearman_corr, spearman_p_value = spearmanr(brent, sent)
plt.title(f'Spearman Correlation between Brent Prices and Sentiment Scores (r={spearman_corr:.2f})')

# Formatting the plot
plt.xlabel('Sentiment Scores')
plt.ylabel('Brent Prices')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Assuming df is your DataFrame
# Calculate the Spearman correlation matrix
spearman_corr = df[['brent', 'WTI', 'SENT']].corr(method='spearman')

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')

# Formatting the plot
plt.title('Spearman Correlation Heatmap')
plt.show()

# Assuming df is your DataFrame
brent = df['brent']
wti = df['WTI']
sent = df['SENT']

# Create a figure with two subplots
plt.figure(figsize=(14, 6))

# Brent vs Sentiment
plt.subplot(1, 2, 1)
sns.regplot(x=sent, y=brent, ci=None, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.xlabel('Sentiment Scores')
plt.ylabel('Brent Prices')
plt.title('Correlation between Brent Prices and Sentiment Scores')
plt.grid(True)

# WTI vs Sentiment
plt.subplot(1, 2, 2)
sns.regplot(x=sent, y=wti, ci=None, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.xlabel('Sentiment Scores')
plt.ylabel('WTI Prices')
plt.title('Correlation between WTI Prices and Sentiment Scores')
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()