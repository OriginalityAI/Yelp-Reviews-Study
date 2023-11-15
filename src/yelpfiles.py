import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import circlify
from matplotlib.ticker import PercentFormatter
from scipy.stats import f_oneway
from IPython.display import clear_output, display
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

chat_dictionary = {'Chat GPT Launch': ['2022-11-30', '#C2002F'],
                   'GPT 3 Launch': ['2020-6-11', '#B0A198'],
                   'GPT 2 Launch': ['2019-2-14', '#7090C2']}


def get_the_dataset():
    """uploading, cleaning and formatting the dataset"""
    df = pd.read_json("../data/dataset_without_text.json")
    # filter out records that did not retrieve aiContent
    df = df[df.aiContent.notna()]
    df['businessOwnerReplies'] = df['businessOwnerReplies'].apply(
        lambda x: True if isinstance(x, list) else False)
    df['elite'] = df['eliteYear'].apply(
        lambda x: True if pd.notna(x) else False)
    df.drop(['previousReviews', 'eliteYear', 'review_id',
            'user_id', 'business_id'], axis=1, inplace=True)
    df['reviewCount'] = df['reviewCount'].fillna(0).astype(int)
    df['friendCount'] = df['friendCount'].fillna(0).astype(int)
    df['stars'] = df['stars'].astype(object)

    def get_reaction(row):
        return max(row['funny'], row['cool'], row['useful'])
    df['reaction'] = df.apply(get_reaction, axis=1)
    df = df[df.aiContent.notna()]

    # convert ai Content to bool
    df['aiContent'] = df.aiContent.apply(lambda x: True if x > 0.5 else False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.reset_index(drop=True)

    display(df.info())
    return (df)


df = get_the_dataset()


def plot_timeline(data, period='M', spanned=8):
    """
    input: dataset(data), period(defaults to 'M'), spanned(defaults to 8 i.e. years)
    output: lineplot of dataset grouped by period over spanned years
    """
    span = data.year.max() - spanned
    data = data[data.year >= span].copy()
    data = data.sort_values('date')
    df = data.groupby(data['date'].dt.to_period(period))['aiContent'].mean()

    fig, ax = plt.subplots(figsize=(12, 9))
    df.plot(ax=ax, color='green')
    for k, v in chat_dictionary.items():
        ax.axvline(v[0], color=v[1], alpha=1, linestyle='--', label=k)

    ax.set_title(
        f"Originality.ai Yelp Reviews Study: \nGrowth of AI Generated Reviews with Time")
    ax.set_xlabel(
        f"{round(len(data)/1000)}K Reviews over {spanned} years plotted monthly")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.legend()
    plot_bars(data)
#     plt.savefig("../images/aiContent_vs_time.png")
    plt.show()

    return data


def stars_experiment(cols, test_col='aiContent', data=df):
    data = data.copy()
    for col in cols:
        tmp = data[[col, test_col]].copy()

        tmp[col] = tmp[col]/tmp[col].max()
        tmp[col] = tmp[col].astype(float)
        model = smf.logit(f'{col} ~ {test_col}', tmp).fit()
        clear_output()
        if col == 'stars':
            ax = sns.barplot(data=data, x=col, y=test_col, hue=col,
                             legend=False,
                             palette='coolwarm', errorbar=None)
            ax.set_title("""Originality.ai Yelp Reviews Study \nAI Content vs Stars (Rating) of Reviews""")
        else:
            pass

        for p in ax.patches:
            height = p.get_height()
            if height < 0.0001:
                continue
            ax.text(p.get_x() + p.get_width() / 2., height,
                    f'{height:.1%}', ha="center", va="bottom")
        
        plt.ylim(0,0.06)
        plt.show()

        print(
            f"Statistical testing indicate that {col} and aiContent are", end=' ')
        print("dependent" if model.pvalues['aiContent[T.True]'] <= 0.05 else
              "independent", end=' ')
        print("features.")
        print("\n\n")


def stats_test_numerical(num_cols, test_col='aiContent', data=df):
    data = data.copy()
    num_subplots = len(num_cols)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 4))
    for idx, col in enumerate(num_cols):
        tmp = data[[col, test_col]].copy()
        tmp = tmp.dropna()
        tmp[col] = tmp[col]/tmp[col].max()
        model = smf.logit(f'{col} ~ {test_col}', tmp).fit()
#         display(tmp)
#         if test_col=='aiContent':
#             tmp['aiContent'].replace({True:'ai', False:'original'}, inplace=True)
#         display(tmp)
        sns.barplot(data=tmp, y=col, x=test_col,
                    hue=test_col, ax=axes[idx])
        axes[idx].get_legend().remove()

        print(
            f"Statistical testing indicate that {col} and aiContent are", end=' ')
        print("dependent" if model.pvalues['aiContent[T.True]'] <= 0.05 else
              "independent", end=' ')
        print("features.")
    
    fig.title("""Originality.ai Yelp Reviews Study \nReview Properties vs AI Content""")
    plt.tight_layout()
    plt.show()


def stats_test_categorical(data, cat_cols, test_col='aiContent'):
    data = data.copy()
    num_subplots = len(cat_cols)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 4))
    for idx, col in enumerate(cat_cols):
        crosstab = pd.crosstab(data[col], data[test_col], normalize='index')
        crosstab.plot(kind="bar", stacked=True, rot=0, ax=axes[idx])

        # Test with chi2_contingency
        contingency = pd.crosstab(data[test_col], data[col], normalize='index')
        stat, p, dof, expected = chi2_contingency(contingency)

        print(f'\n{col} - p-value is {p:.2f}')
        print(f'aiContent and {col} are probably',
              'independent.' if p > 0.05 else 'dependent.')
    plt.show()


def get_monthly_average_and_average(key, period, data):
    ts = data.loc[period[1]:period[0]]
    time_diff = period[0] - period[1]
    x = period[1] + (time_diff / 2)
    ts = ts.reset_index()
    ts = ts.groupby(ts['date'].dt.to_period('M'))['aiContent'].mean()
    return x, ts.mean(), period[0]-period[1]


def plot_bars(data):
    data = data.sort_values('date')[['date', 'aiContent']].set_index('date')

    launch_dates = {k: pd.to_datetime(v[0])
                    for k, v in chat_dictionary.items()}
    latest_date, earliest_date = data.index[-1], data.index[0]

    periods = {}
    for k, v in launch_dates.items():
        if k == 'Chat GPT Launch':
            periods[k] = (latest_date, v)
        elif k == 'GPT 3 Launch':
            periods[k] = (launch_dates['Chat GPT Launch'] -
                          pd.DateOffset(days=1), v)
        elif k == 'GPT 2 Launch':
            periods[k] = (launch_dates['GPT 3 Launch'] -
                          pd.DateOffset(days=1), v)
    periods['pre-GPT Launch'] = (v - pd.DateOffset(days=1), earliest_date)

    color_bars = [v[1] for v in chat_dictionary.values()] + ['#3C4E92']

    for i, (key, period) in enumerate(periods.items()):

        x, height, width = get_monthly_average_and_average(key, period, data)
        plt.bar(x, height, width, align='center', alpha=0.1,
                color=color_bars[i % len(color_bars)])
        h = 0.02
        plt.text(x, h, f'{height*100:.1f}%', ha='center',
                 va='bottom', fontsize=16, fontweight='bold')
