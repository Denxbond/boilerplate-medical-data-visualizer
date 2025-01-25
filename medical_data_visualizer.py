import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data():
    # Import data
    df = pd.read_csv('medical_examination.csv')

    # Add 'overweight' column
    df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

    # Normalize data by making 0 good and 1 bad
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    return df

def draw_cat_plot(df):
    # Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group and reformat the data to split by 'cardio'
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Draw the catplot
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    ).fig

    return fig

def draw_heat_map(df):
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        cbar_kws={'shrink': .5},
        ax=ax
    )

    return fig

if __name__ == "__main__":
    df = load_and_process_data()

    # Generate and save the categorical plot
    cat_plot = draw_cat_plot(df)
    cat_plot.savefig('catplot.png')

    # Generate and save the heat map
    heat_map = draw_heat_map(df)
    heat_map.savefig('heatmap.png')
