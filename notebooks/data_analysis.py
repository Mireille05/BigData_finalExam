import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

# Define paths based on your directory structure
base_path = r'C:\Users\CTRL-SHIFT LTD\Desktop\BigData_Exam'
plant_fact_path = os.path.join(base_path, 'data', 'raw', 'Plant_DTS.xls')
accounts_path = os.path.join(base_path, 'data', 'raw', 'AccountAnalysed.xlsx')
plant_hierarchy_path = os.path.join(base_path, 'data', 'raw', 'Plant_Hearchy.xlsx')
output_cleaned = os.path.join(base_path, 'data', 'cleaned')
output_enhanced = os.path.join(base_path, 'data', 'enhanced')
output_screenshots = os.path.join(base_path, 'powerbi', 'screenshots')

# Create output directories
os.makedirs(output_cleaned, exist_ok=True)
os.makedirs(output_enhanced, exist_ok=True)
os.makedirs(output_screenshots, exist_ok=True)

# Verify file existence
for path in [plant_fact_path, accounts_path, plant_hierarchy_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load data
try:
    df_fact = pd.read_excel(plant_fact_path, sheet_name='Plant_FACT', engine='xlrd')
    df_accounts = pd.read_excel(accounts_path)
    df_hierarchy = pd.read_excel(plant_hierarchy_path)
except ImportError:
    raise ImportError("Missing 'xlrd' package. Install it using 'pip install xlrd>=2.0.1'")

# Merge datasets
df = df_fact.merge(df_accounts, on='Account_id', how='left')
df = df.merge(df_hierarchy, left_on='Product_id', right_on='Product_Name_id', how='left')

# Function to clean data
def clean_data(df):
    print(f"Rows before dropping NA: {len(df)}")
    df = df.dropna()
    print(f"Rows after dropping NA: {len(df)}")
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
    Q1 = df['quantity'].quantile(0.25)
    Q3 = df['quantity'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['quantity'] < (Q1 - 1.5 * IQR)) | (df['quantity'] > (Q3 + 1.5 * IQR)))]
    return df

# Function to enhance data
def enhance_data(df):
    df['Sales_Per_Unit'] = df['Sales_USD'] / df['quantity'].replace(0, np.nan)
    if 'country_code' in df.columns:
        df = pd.get_dummies(df, columns=['country_code'], prefix='country_code')
    scaler = StandardScaler()
    numeric_cols = [col for col in ['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD'] if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Function for EDA
def run_eda(df):
    print("Descriptive Statistics:")
    numeric_cols = [col for col in ['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD', 'Sales_Per_Unit'] if col in df.columns]
    if numeric_cols:
        stats = df[numeric_cols].describe()
        print(stats)
        print("\nAdditional Statistics:")
        print(f"Mean of Quantity: {df['quantity'].mean():.2f}")
        print(f"Median of Quantity: {df['quantity'].median():.2f}")
        print(f"Mode of Quantity: {df['quantity'].mode()[0] if not df['quantity'].mode().empty else 'N/A'}")
        print(f"Skewness of Quantity: {df['quantity'].skew():.2f}")

    # Box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['quantity'])
    plt.title('Box Plot of Quantity')
    plt.ylabel('Quantity')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_screenshots, 'quantity_boxplot.png'), bbox_inches='tight')
    plt.close()

    # Histogram with KDE
    plt.figure(figsize=(8, 5))
    sns.histplot(df['quantity'], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title('Quantity Distribution with KDE')
    plt.xlabel('Quantity')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_screenshots, 'quantity_dist_kde.png'), bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    if numeric_cols:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(output_screenshots, 'correlation_heatmap.png'), bbox_inches='tight')
        plt.close()

    # Sales vs Quantity by country
    country_cols = [col for col in df.columns if col.startswith('country_code_')]
    if country_cols and 'Sales_USD' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Sales_USD', y='quantity', hue=country_cols[0], data=df, palette='tab10', alpha=0.7)
        plt.title(f'Sales USD vs Quantity by {country_cols[0]}')
        plt.xlabel('Sales (USD)')
        plt.ylabel('Quantity')
        plt.grid(True)
        plt.savefig(os.path.join(output_screenshots, 'sales_vs_qty_country.png'), bbox_inches='tight')
        plt.close()
    else:
        print("No country code columns or Sales_USD found for scatter plot.")

# Function for clustering
def train_model(df):
    X = df[['quantity', 'Sales_USD']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k_values = range(2, 11)
    silhouette_scores = []
    inertias = []
    best_k = 2
    best_score = -1
    best_kmeans = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

    print(f"Best k: {best_k}, Silhouette Score: {best_score:.4f}")
    df['Cluster'] = best_kmeans.fit_predict(X_scaled)

    # Plot Elbow and Silhouette
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_screenshots, 'elbow_silhouette.png'), bbox_inches='tight')
    plt.close()

    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='Set2')
    plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('KMeans Clustering (Sales vs Quantity)')
    plt.xlabel('Scaled Sales USD')
    plt.ylabel('Scaled Quantity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_screenshots, 'kmeans_scaled.png'), bbox_inches='tight')
    plt.close()

    return df

# Main execution
if __name__ == "__main__":
    # Clean data
    cleaned_df = clean_data(df)
    cleaned_df.to_excel(os.path.join(output_cleaned, 'cleaned_plant_fact.xlsx'), index=False)

    # Enhance data
    enhanced_df = enhance_data(cleaned_df)
    enhanced_df.to_excel(os.path.join(output_enhanced, 'enhanced_plant_fact.xlsx'), index=False)

    # Run EDA
    run_eda(cleaned_df)

    # Train and evaluate model
    final_df = train_model(enhanced_df)
    final_df.to_excel(os.path.join(output_enhanced, 'clustered_plant_fact.xlsx'), index=False)

    # Additional Power BI visualizations
    df_enhanced = pd.read_excel(os.path.join(output_enhanced, 'enhanced_plant_fact.xlsx'))
    df_enhanced.columns = df_enhanced.columns.str.strip()

    # Price per Unit vs Quantity
    country_cols = [col for col in df_enhanced.columns if col.startswith('country_code_')]
    if country_cols and 'Sales_Per_Unit' in df_enhanced.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_enhanced, x='Sales_Per_Unit', y='quantity', hue=country_cols[0], palette='tab10', alpha=0.7)
        plt.title('Price per Unit vs Quantity')
        plt.xlabel('Price per Unit (USD)')
        plt.ylabel('Quantity Sold')
        plt.grid(True)
        plt.savefig(os.path.join(output_screenshots, 'price_vs_quantity.png'), bbox_inches='tight')
        plt.close()

    # Total Quantity Sold per Country
    if 'country2' in df_enhanced.columns:
        plt.figure(figsize=(10, 6))
        country_summary = df_enhanced.groupby('country2')['quantity'].sum().sort_values(ascending=False)
        sns.barplot(x=country_summary.values, y=country_summary.index, palette='viridis')
        plt.title('Total Quantity Sold per Country')
        plt.xlabel('Quantity')
        plt.ylabel('Country')
        plt.savefig(os.path.join(output_screenshots, 'quantity_per_country.png'), bbox_inches='tight')
        plt.close()
    else:
        print("No 'country2' column found for quantity per country plot.")

    print("✅ All plots saved successfully in 'powerbi/screenshots/' 🎉")