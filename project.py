import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_dataset(filepath="diwali_sales_data.csv"):

    try:
        df = pd.read_csv(filepath)
        # dropping the 'User_ID' column
        df = df.drop(['User_ID'], axis=1)
        df = df.drop(['Status'], axis =1)
        df = df.drop(['unnamed1'],axis=1)
        
        df = df.drop_duplicates()
        print(f"The dimensions of the imported dataset: {df.shape}")
        return df
    
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file. Check the file format and encoding.")
        return None

def cleanse_dataset(df):

    """
    Clean the dataset:
      Remove rows with missing or irrelevant values.
      Ensure data transformations and replace values as needed.
    """

    if df is None:
        print("No dataset provided for cleansing.")
        return None
    
    initial_row_count = df.shape[0]
    
    # Drop rows with missing values in crucial columns
    df = df.dropna(subset=['Amount', 'Orders', 'Age'])
    
    # Type coercion with error handling
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Remove invalid values (using numpy for better handling)
    df = df[(df['Amount'] > 0) & (df['Age'] > 0)]
    
   
   
    # Replace missing values in categorical columns with 'Unknown' or a suitable default
    fill_values = {
        'Product_Category': 'Unknown',
        'State': 'Unknown',
        'Zone': 'Unknown',
        'Occupation': 'Not specified',
        'Marital_Status': 'Unknown',
        'Gender': 'Not specified'
    }

    df.fillna(value=fill_values, inplace=True)


    # Replace outliers in 'Age' with NaN and fill them with the median age
    median_age = np.median(df['Age'])
    df['Age'] = np.where(df['Age'] > 100, np.nan, df['Age'])  
    df['Age'].fillna(median_age, inplace=True)

    final_row_count = df.shape[0]
    print(f"Rows dropped during cleansing: {initial_row_count - final_row_count}")
    
    return df

def plot_gender_vs_total_amount(df):
   
    # Calculate total amount spent by gender
    gender_totals = df.groupby('Gender')['Amount'].sum().reset_index()

    # Plot the bar chart with a custom color palette
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Gender', y='Amount', data=gender_totals, palette='viridis')

    # the total amounts on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', 
                    (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=12)

    plt.title('Total Amount Spent by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Total Amount Spent', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()



def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=10, kde=True)

    plt.title('Age Distribution of Customers', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of Customers', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_state_wise_sales(df):

    plt.figure(figsize=(12, 7))

    state_sales = df.groupby('State')['Amount'].sum().sort_values(ascending=False)

    colors = plt.cm.viridis(np.linspace(0, 1, len(state_sales)))

    # bar chart
    plt.barh(state_sales.index, state_sales.values, color=colors)


    plt.title('Total Sales by State', fontsize=16)
    plt.xlabel('Total Amount', fontsize=14)
    plt.ylabel('State', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_product_category_vs_amount(df):

        plt.figure(figsize=(12, 7))
        sns.boxplot(x='Product_Category', y='Amount', data=df, palette='Set2')


        plt.title('Amount Spent by Product Category', fontsize=16)
        plt.xlabel('Product Category', fontsize=14)
        plt.ylabel('Amount Spent', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)  # Rotate x-tick labels for better visibility
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.show()


def plot_marital_status_vs_amount(df):

    df['Marital_Status'] = df['Marital_Status'].map({0: 'Unmarried', 1: 'Married'})

    marital_status_totals = df.groupby('Marital_Status')['Amount'].sum().reset_index()

    married_df = df[df['Marital_Status'] == 'Married']
    unmarried_df = df[df['Marital_Status'] == 'Unmarried']

    married_gender_totals = married_df.groupby('Gender')['Amount'].sum().reset_index()
    unmarried_gender_totals = unmarried_df.groupby('Gender')['Amount'].sum().reset_index()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.barplot(x='Marital_Status', y='Amount', data=marital_status_totals, palette='Set2')
    plt.title('Total Amount Spent by Marital Status', fontsize=14)
    plt.xlabel('Marital Status', fontsize=12)
    plt.ylabel('Total Amount Spent', fontsize=12)
    for i, row in enumerate(marital_status_totals.itertuples()):
        plt.text(i, row.Amount, f'{row.Amount:,.0f}', ha='center', va='bottom', fontsize=10)

    plt.subplot(1, 3, 2)
    sns.barplot(x='Gender', y='Amount', data=married_gender_totals, palette='viridis')
    plt.title('Amount Spent by Married Men and Women', fontsize=14)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Total Amount Spent', fontsize=12)
    for i, row in enumerate(married_gender_totals.itertuples()):
        plt.text(i, row.Amount, f'{row.Amount:,.0f}', ha='center', va='bottom', fontsize=10)

    plt.subplot(1, 3, 3)
    sns.barplot(x='Gender', y='Amount', data=unmarried_gender_totals, palette='coolwarm')
    plt.title('Amount Spent by Unmarried Men and Women', fontsize=14)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Total Amount Spent', fontsize=12)
    for i, row in enumerate(unmarried_gender_totals.itertuples()):
        plt.text(i, row.Amount, f'{row.Amount:,.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_stacked_bar_and_pie(df):
    
    occupation_orders = df.groupby('Occupation')['Orders'].sum().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    occupation_orders.plot(kind='bar', stacked=True, color=['lightblue', 'orange', 'lightgreen', 'purple'], ax=axes[0])
    axes[0].set_title('Total Orders by Occupation (Stacked Bar Chart)', fontsize=16)
    axes[0].set_xlabel('Occupation', fontsize=14)
    axes[0].set_ylabel('Total Orders', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(occupation_orders, labels=occupation_orders.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=90)
    axes[1].set_title('Total Orders by Occupation (Pie Chart)', fontsize=16)
    
    plt.tight_layout()
    plt.show()


def plot_zone_vs_total_amount(df):

    zone_totals = df.groupby('Zone')['Amount'].sum()
    
    plt.figure(figsize=(10, 7))
    plt.pie(zone_totals, labels=zone_totals.index, autopct='%1.1f%%', colors=plt.cm.tab20.colors, startangle=140)
    plt.title('Total Amount Spent by Zone', fontsize=16)
    plt.show()


def plot_gender_vs_avg_order_value(df):
    """Plot a bar plot showing the average order value by gender with error bars."""
    avg_order_value = df.groupby('Gender')['Amount'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Gender', y='Amount', data=df, estimator=np.mean, ci='sd', palette='pastel')
    plt.title('Average Order Value by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Average Order Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot_age_vs_amount(df):

    age_amount = df.groupby('Age')['Amount'].sum().sort_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(age_amount.index, age_amount.values, marker='o', linestyle='-', color='darkorange')
    plt.title('Age vs. Amount Spent', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Total Amount Spent', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_top_10_product_categories(df):
 
    product_sales = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(14, 7))
    product_sales.plot(kind='barh', color='slateblue')
    plt.title('Top 10 Product Categories by Sales', fontsize=16)
    plt.xlabel('Total Sales Amount', fontsize=14)
    plt.ylabel('Product Category', fontsize=14)
    plt.gca().invert_yaxis()  # Invert the y-axis for better visibility
    plt.show()

def plot_order_frequency_by_age_group(df):

    # Categorize age into bins
    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    age_order_counts = df.groupby('Age_Group')['Orders'].sum().reset_index()

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Age_Group', y='Orders', data=age_order_counts, palette='YlGnBu')
    plt.title('Order Frequency by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Total Number of Orders', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def plot_occupation_vs_zone_sales(df):

    zone_occupation_sales = df.groupby(['Zone', 'Occupation'])['Amount'].sum().unstack()
    zone_occupation_sales.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20')
    plt.title('Sales by Occupation across Zones', fontsize=16)
    plt.xlabel('Zone', fontsize=14)
    plt.ylabel('Total Sales Amount', fontsize=14)
    plt.legend(title='Occupation', bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.show()

def plot_repeat_orders(df):

    repeat_customers = df.groupby('Cust_name')['Orders'].count().reset_index()
    repeat_customers['Repeat'] = repeat_customers['Orders'] > 1
    repeat_counts = repeat_customers['Repeat'].value_counts()
    repeat_counts.plot.pie(autopct='%1.1f%%', labels=['Single Purchase', 'Repeat Purchaser'], startangle=140)
    plt.title('repeat order Analysis', fontsize=16)
    plt.show()

def plot_avg_order_value_by_age_group(df):

    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    avg_order_value = df.groupby('Age_Group')['Amount'].mean().reset_index()
    sns.barplot(x='Age_Group', y='Amount', data=avg_order_value, palette='magma')
    plt.title('Average Order Value by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Average Order Value', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()



def analyze_occupation_gender_product_category(df):
 
    occupation_gender_product = df.groupby(['Occupation', 'Gender', 'Product_Category'])['Amount'].sum().reset_index()

    pivot_table = occupation_gender_product.pivot_table(
        index='Occupation',
        columns=['Gender', 'Product_Category'],
        values='Amount',
        aggfunc='sum'
    )
    
    # Plot a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=False, cmap='coolwarm', cbar_kws={'label': 'Total Amount Spent'})
    
    # Customize the plot
    plt.title('Purchasing Behavior by Occupation, Gender, and Product Category', fontsize=16)
    plt.xlabel('Gender and Product Category', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_occupation_vs_state_area(df):
   
#stacked area chart showing occupation-based spending patterns across states.

    state_occupation_sales = df.groupby(['State', 'Occupation'])['Amount'].sum().unstack().fillna(0)
    
    state_occupation_sales = state_occupation_sales.loc[state_occupation_sales.sum(axis=1).sort_values(ascending=False).index]
    
    ax = state_occupation_sales.plot(kind='area', stacked=True, colormap='tab20', alpha=0.8, figsize=(14, 8))
    
    # plot
    plt.title('Occupation-Based Spending Patterns Across States', fontsize=16)
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Total Sales Amount', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_age_vs_amount(df):
    sns.scatterplot(x='Age', y='Amount', data=df, hue='Gender', alpha=0.7)
    plt.title('Age vs. Amount')
    plt.xlabel('Age')
    plt.ylabel('Amount Spent')
    plt.grid(True)
    plt.show()



def main():
    # Load and cleanse the dataset
    dataset = read_dataset()
    if dataset is not None:
        cleaned = cleanse_dataset(dataset)
        if cleaned is not None and not cleaned.empty:
            print("Preview of cleaned data:")
            print(cleaned.head())
            
          
            plot_gender_vs_total_amount(cleaned)
            plot_age_distribution(cleaned)
            plot_state_wise_sales(cleaned)
            plot_product_category_vs_amount(cleaned)
            plot_marital_status_vs_amount(cleaned)
            plot_stacked_bar_and_pie(cleaned)
            plot_zone_vs_total_amount(cleaned)
            plot_gender_vs_avg_order_value(cleaned)
            plot_age_vs_amount(cleaned)
            plot_top_10_product_categories(cleaned)
            plot_order_frequency_by_age_group(cleaned)
            plot_occupation_vs_zone_sales(cleaned)
            plot_repeat_orders(cleaned)
            # plot_avg_order_value_by_age_group(cleaned)
            plot_occupation_vs_state_area(cleaned)
            # analyze_occupation_gender_product_category(cleaned)

          


        else:
            print("The dataset is empty after cleansing.")
    else:
        print("Failed to load or clean dataset.")

if __name__ == "__main__":
    main()