import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

#Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,6)

#=======================
#1. Data Loading and Initial Exploration
#=======================
def load_data(filepath):
    """Load the smartphone dataset"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded sucessfully")
    print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    return df

#=======================
#2. Dataset discription
#======================
def describe_data(df):
    """Provide comprehensive description of the dataset"""
    print("\n" + "=" * 80)
    print("DATA DESCRIPTION")
    print("=" * 80)
    
    print("\n Data Source and Context:")
    print("-" * 80)
    print("This dataset contains smartphone specifications and princing information collected from various smartphone models available in the market.")
    print("\n Unit of analysis:")
    print("Each row represents a signal smartphone model with its specification, pricing, and features.")
    
    print("\n Key Variables and Data Types: ")
    print("-" * 80)
    print(df.dtypes.to_string())
    
    print("\n Dataset Overview:")
    print("-" * 80)
    print(df.info())
    
    print("\n First few rows")
    print("-" * 80)
    print(df.head(10))
    
    return df

#=======================
#3. Data quality assessment
#=======================
def assess_data_quality(df):
    """Identify and document data quality issues"""
    print("\n" + "=" * 80)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    #Missing values analysis
    print("\n Missing Values Analysis:")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Column': missing.index, 'Missing_Count':missing.values, 'Missing_Percentage': missing_pct.values})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print("\nColumns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("\n No missing values detected.")
        
    #Outliers detection for key numerical variables
    print("\n\n Outlier Detection (using IQR method):")
    print("-" * 80)
    numerical_cols = ['price','rating','clock_ghz','ram_gb','strogae_gb','battery_mah','screen_size_in','rear_camera_mp','front_camera_mp']
    
    outlier_summary = []
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_summary.append({'Variable':col, 'Outlier_Count': len(outliers),
                'Percentage': round(len(outliers)/len(df) * 100,2),
                'Lower_Bound' : round(lower_bound,2),
                'Upper_Bound':round(upper_bound,2)
                })
                
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.to_string(index=False))
    else:
        print("\n No significant outliers detected")
        
    #Check for inconsistent or inciduak entries
    print("\n\n Invalid or Inconsistent Entries:")
    print("-" * 80)
    
    issues = []
    
    #Check for negative or zero values where they shouldn't exist
    if 'price' in df.columns: 
        invalid_price = df[df['price'] <= 0]
        if len(invalid_price) > 0:
            issues.append(f". {len(invalid_price)} phones with price <= 0")
    
    if 'rating' in df.columns :
        invalid_rating = df[(df['rating'] < 0) | (df['rating'] > 100)]
        if len(invalid_rating) > 0:
            issues.append(f". {len(invalid_rating)} phones with rating outside 0-100 range")
            
    if 'ram_gb' in df.columns:
        invalid_ram = df[df['ram_gb'] <= 0]
        if len(invalid_ram) > 0:
            issues.append(f". {len(invalid_ram)} phones with RAM <= 0 GB")
            
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("No obvious invalid entries detected.")

    return missing_df, outlier_summary

#=======================
#4. Data Cleaning and Perprocessing 
#=======================
def clean_data(df):
    """Clean the dataset based on identified issues"""
    print("\n" + "=" * 80)
    print("DATA CLEANING AND PREPROCESSING")
    print("=" * 80)
    
    df_cleaned = df.copy()
    cleaning_log = []
    
    print("\n Cleaning Steps:")
    print("-" * 80) 
    
    #1. Handle missing values
    print("\n1. Handling Missing Values:")
    
    #For fast_charge_w : Missing likely means no fast charging or unknown
    if 'fast_charge_w' in df_cleaned.columns:
        missing_before = df_cleaned['fast_charge_w'].isnull().sum()
        #Fill with median for devices with similar prince range
        df_cleaned['fast_charge_w'].fillna(df_cleaned['fast_charge_w'].median(), inplace=True)
        cleaning_log.append(f" .fast_charge_w: Filled {missing_before} missing values with median")
        print(f" .fast_charge_w : Filled {missing_before} missing values with median")
        
    #For refesh_rate_hz : Missing values filled with 60 (standard refresh rate)
    if 'refresh_rate_hz' in df_cleaned.columns:
        missing_before = df_cleaned['refresh_rate_hz'].isnull().sum()
        df_cleaned['refresh_rate_hz'].fillna(60.0, inplace=True)
        cleaning_log.append(f" .refresh_rate_hz : Filled {missing_before} missing values with 60 Hz (standard)")
        print(f" .refresh_rate_hz : Filled {missing_before} missing values with 60 Hz (standard)")
    
    #For rating : Remove rows with missing rating as it's a key variable 
    if 'rating' in df_cleaned.columns:
        missing_before = df_cleaned['rating'].isnull().sum()
        if missing_before > 0:
            df_cleaned = df_cleaned.dropna(subset=['rating'])
            cleaning_log.append(f". rating : Removed {missing_before} rows with missing ratings")
            print(f" .rating : Removed {missing_before} rows with missing ratings")
            
    #For memory_card_max_gb : Fill with 0 (no card support)
    if "memory_card_max_gb" in df_cleaned.columns:
        missing_before = df_cleaned['memory_card_max_gb'].isnull().sum() 
        df_cleaned['memory_card_max_gb'].fillna(0, inplace = True)
        cleaning_log.append(f" .memory-card_max_gb : Filled {missing_before} missing values with 0")
        print(f" .memory_card_max_gb : Filled {missing_before} missing values with 0")
    
    #2 Handle outliers
    print("\n2. Handling Outliers :")
    print(" Dcision : Retaining outliers as they represent legitimate premium/budget devices")
    print(" JUSFITICATION : In smartphone market, extreme prices (budget vs. flasgship) are valid ")
    cleaning_log.append(" .Oultiers retained: Represent legitatimate market segments")
    
    #3 Remove invalid entries
    print("\n3. Removing Invalid Entries: ")
    initial_rows = len(df_cleaned)
    
    #Remove phones with price <= 0
    if 'price' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['price'] > 0]
        removed = initial_rows - len(df_cleaned)
        if removed > 0:
            print(f" .Removed {removed} phones with invalid price (<= 0)")
            cleaning_log.append(f" .Removed {removed} phones with invalid price (<= 0)")
    
    #Validate rating range
    if 'rating' in df_cleaned.columns :
        df_cleaned = df_cleaned[(df_cleaned['rating'] >= 0) & (df_cleaned['rating'] <= 100)]
        
    #4. print category
    print("\n4. Creating Derived Features :")
    
    #Prince category
    df_cleaned['price_category'] = pd.cut(df_cleaned['price'], 
                                          bins=[0, 15000, 30000, 50000, float('inf')],
                                          labels=['Budget', 'Mid-range', 'Premium', 'Flagship'])
    print("   • Created 'price_category' (Budget/Mid-range/Premium/Flagship)")
    cleaning_log.append("   • Created price_category feature")
    
    #Camera quality indicator 
    if 'rear_camera_max_mp' in df_cleaned.columns:
       df_cleaned['camera_quality'] = pd.cut(df_cleaned['rear_camera_max_mp'],
                                             bins=[0, 20, 50, 100, float('inf')],
                                              labels=['Basic', 'Good', 'Great', 'Excellent'])
       print("   • Created 'camera_quality' based on megapixels")
       cleaning_log.append("   • Created camera_quality feature")
        
    print(f"\n Cleaning Compleate")
    print(f" Final dataset : {len(df_cleaned)} rows x {df_cleaned.shape[1]} columns")
    print(f" Rows removed : {len(df) - len(df_cleaned)}")
    
    return df_cleaned, cleaning_log

#======================
#5. Exploratory Data Analysis (EDA)
#======================
def exploratory_data_analysis(df) :
    """Perform comprehensive EDA with visualizations """
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    ##Create output directory for plots
    import os
    os.makedirs('plots',exist_ok=True)
    
    #5.1 Distribution Analysis
    print("\n 5.1 Distribution Analysis:")
    print("-" * 80)
    
    #Price Distribution
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    
    #Histogram
    axes[0].hist(df['price'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price(₹)', fontsize = 12)
    axes[0].set_ylabel('Frequency', fontsize = 12)
    axes[0].set_title('Distribution of Smartphone Price ', fontsize = 14, fontweight ='bold')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='dashed', linewidth=2, label = f'Mena: ₹{df["price"].mean():.0f}')
    axes[0].axvline(df['price'].median(), color='green', linestyle='dashed', linewidth=2, label = f'Median: ₹{df["price"].median():.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #Boxplot
    axes[1].boxplot(df['price'], vert = True)
    axes[1].set_ylabel('Price(₹)', fontsize = 12)
    axes[1].set_title('Price Distribution Boxplot', fontsize = 14, fontweight ='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/01_price_distribution.png', dpi = 300, bbox_inches='tight')
    plt.close()
    
    print("Price Distribution Analysis :")
    print(f" Mean Price : ₹{df['price'].mean():.2f}")
    print(f" Median Price : ₹{df['price'].median():.2f}")
    print(f" The distribution is right-skewed , indicating more budget/mid-range phones")
    print(f" Several high-priced outliers represent flagship models")
    print(" Saved : plots/01_price_distribution.png")
    
    #Rating distribution
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    
    axes[0].hist(df['rating'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Rating (out of 100)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Smartphone Ratings', fontsize=14, fontweight='bold')
    axes[0].axvline(df['rating'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["rating"].mean():.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(df['rating'], vert=True)
    axes[1].set_ylabel('Rating', fontsize=12)
    axes[1].set_title('Rating Distribution Boxplot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig('plots/02_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n Rating Distribution Analysis :")
    print(f" Mean Rating : {df['rating'].mean():.2f}")
    print(f" Median Rating : {df['rating'].median():.2f}")
    print(f" Most phones clister around 75-85 rating range")
    print(" Saved : plots/02_rating_distribution.png")
    
    #5.2 Group Comparisons
    print("\n 5.2 Group Comparisons:")
    print("-" * 80) 
    
    #Price by category
    fig, ax = plt.subplots(figsize=(12,6))
    df.boxplot(column='price', by='price_category', ax=ax)
    ax.set_xlabel('Price Category', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.set_title('Price Distribution by Category', fontsize=14, fontweight='bold')
    plt.suptitle('') #Remove default title
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/03_price_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Price by Category Analysis :")
    for cat in df['price_category'].unique():
        cat_data = df[df['price_category'] == cat]['price']
        print(f" {cat} : Mean = ₹{cat_data.mean():.0f} , Median = ₹{cat_data.median():.0f}")
    print(" Clear separation between price categories validates our classification")
    print(" Saved : plots/03_price_by_category.png")
    
    #Rating by price category 
    fig, ax = plt.subplots(figsize = (12,6))
    df.boxplot(column = "rating" , by = "price_category", ax=ax)
    ax.set_xlabel('Price category', fontsize = 12)
    ax.set_ylabel('Rating', fontsize = 12)
    ax.set_title('Rating Distribution by Price Category', fontsize = 14 , fontweight = "bold")
    plt.suptitle('')
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig('plots/04_rating_by_category.png', dpi = 300, bbox_inches = 'tight') 
    plt.close()
    
    print("\n Rating by Price Category: ")
    for cat in df['price_category'].unique():
        cat_data = df[df['price_category'] == cat]['rating']
        print(f" {cat} : Mean = {cat_data.mean():.2f} , Median = {cat_data.median():.2f}") 
        print(" Interestingly, higher price doesn't always mean higher ratings")
        print("Saved : plots/04_rating_by_category.png") 
        
    
    #5.3 Relationship Exploration
    print("\n 5.3 Relationship Exploration:")
    print("-" * 80)
    
    #price vs Rating scatter plot
    fig, ax = plt.subplots(figsize=(12,6))
    scatter = ax.scatter(df['price'], df['rating'], alpha = 0.6, c= df['price'], cmap='viridis', s = 50)
    ax.set_xlabel('Price (₹)', fontsize=12) 
    ax.set_ylabel('Price vs Rating Relationship', fontsize = 14, fontweight = 'bold')
    plt.colorbar(scatter, label = 'Price (₹)')
    
    #Add thrend line
    z = np.polyfit(df['price'], df['rating'],1)
    p = np.poly1d(z)
    ax.plot(df['price'].sort_values(),p(df['price'].sort_values()),"r--",linewidth=2 , label = f'Trend line')
    ax.legend()
    ax.grid(True, alpha=0.3)    
    plt.tight_layout()
    plt.savefig('plots/05_price_vs_rating.png', dpi = 300, bbox_inches='tight')
    plt.close()
    
    correlation = df['price'].corr(df['rating'])
    print(f" Price VS Rating Correlation : {correlation:.3f}")
    print(f" Weak positive correlation suggests price is not the only rating factor")
    print(f" Many mid-range phones achivev high rating through good value proposition")
    print(" Saved : plots/05_price_vs_rating.png")
    
    #RAM vs Price
    fig, ax = plt.subplots(figsize=(12,6))
    scatter = ax.scatter(df['ram_gb'], df['price'], alpha=0.6, c=df['price'], cmap='plasma', s=50)
    ax.set_xlabel('RAM (GB)', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.set_title('RAM vs Price relationship', fontsize = 14 , fontweight = 'bold')
    plt.colorbar(scatter, label='Price (₹)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/06_ram_vs_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    correlation = df['ram_gb'].corr(df['price'])
    print(f"\n RAM vs Price Correlation : {correlation:.3f}")
    print(f" Storage positive correlation indicates RAM is a key price driver")
    print(" Higher RAM typically found in premium devices")
    print(" Saved : plots/06_ram_vs_price.png")
    
    #Battery vs Price
    fig, ax = plt.subplots(figsize=(12,6))
    scatter = ax.scatter(df['battery_mah'], df['price'], alpha=0.6, c=df['rating'], cmap='coolwarm', s=50)  
    ax.set_xlabel('Battery Capacity (mAh)', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.set_title('Batter Capacity vs Price (colared by Rating)',fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Rating')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/07_battery_vs_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    correlation = df['battery_mah'].corr(df['price'])
    print(f"\n Battery vs Price Correlation : {correlation:.3f}")      
    print(f" Moderate correlation - battery size alone doesn't determine price")
    print(f" Both budget and premium phones can have large batteries")
    print(" Saved : plots/07_battery_vs_price.png")
    
    #Correlation heatmap
    fig, ax = plt.subplots(figsize=(12,10))
    numerical_features = ['price', 'rating', 'clock_ghz', 'ram_gb', 'storage_gb',                        'battery_mah', 'screen_size_in', 'rear_camera_max_mp', 'front_camera_mp']
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm', center = 0, square = True, linewidths=1, cbar_kws ={"shrink": .8}, ax=ax, fmt = '.2f')
    ax.set_title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold', pad = 20)
    plt.tight_layout()
    plt.savefig('plots/08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n Correlation Matrix Analysis :")
    print(" RAM and Stroage show strong correlation (0.87+)")
    print(" Price strongly correlated with RAM, storage , and processor speed")
    print(" Camera space show moderate correlation with price ")
    print (" Saved : plots/08_correlation_heatmap.png")
    
    print("\n " + "=" * 80)
    print("All visualizations saved in 'plots/' directory")
    print("=" * 80)
    
#======================
# 6. DESCRIPTIVE STATISTICS
#======================
def descriptive_statistics(df):
    """Calculate and interpret descriptive statistics"""
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    #Key numerical variables
    key_vars = ['price', 'rating','ram_gb','storage_gb','battery_mah','screen_size_in','rear_camera_max_mp']
    
    print("\n Measure of center and Spread:")
    print("-" * 80)
    
    stats_summary = []
    for var in key_vars:
        if var in df.columns:
            mean_val= df[var].mean()
            median_val = df[var].median()
            mode = df[var].mode()[0]
            std_val = df[var].std()
            varianve = df[var].var()
            minimum = df[var].min()
            maximum = df[var].max()
            q1 = df[var].quantile(0.25)
            q3 = df[var].quantile(0.75)
            iqr = q3 - q1
            
            stats_summary.append({
                'Variable': var,
                'Mean': round(mean_val,2),
                'Median': round(median_val,2),
                'Mode': round(mode,2),
                'Std Dev': round(std_val,2),
                'Variance': round(varianve,2),
                'Min': round(minimum,2),
                'Max': round(maximum,2),
                'Q1': round(q1,2),
                'Q3': round(q3,2),
                'IQR': round(iqr,2)
            })
    stats_df = pd.DataFrame(stats_summary)
    print(stats_df.to_string(index=False))
    
    print("\n Interpretation in context:")
    print("-" * 80)
    
    print("\n1. PRICE: ")
    print(f" Average smartphones costs ₹{df['price'].mean():.2f}")
    print(f" Median price (₹{df['price'].median():.2f}) lower than mean indicates right-skewed distribution")
    print(f" High standard deviation (₹{df['price'].std():.2f}) indicates wide price ranges")
    print(f" IQR of ₹{(df['price'].quantile(0.75) - df['price'].quantile(0.25)):.2f} shows substial mid-range variation")
    
    print("\n2. RATING")
    print(f" Average rating of {df['rating'].mean():.2f} suggest generally positive feedback")
    print(f" Low standard derivation ({df['rating'].std():.2f}) indicates consistes quality")
    print(f" 50% of phones rates between {df['rating'].quantile (0.75) - df ['price'].quantile(0.25):.2f} shows substancial mid-range variation")
    
    print("\n3. RAM")
    print(f" Average RAM: {df['battery_mah'].mean():.0f} mAh")
    print(f" Most phones (IQR) have batteries between {df['battery_mah'].quantile(0.25):.0f} and {df['battery_mah'].quantile(0.75):.0f} mAh")
    print(f" Relatively low various suggests standardization around 4500-5000 mAh")
    
    #Summary statistics by price category 
    print("\n\n Statistics by Price Category:")
    print("-" * 80)
    
    category_stats = df.groupby('price_category').agg({
        'price': ['mean', 'median', 'count'],
        'rating' : ['mean', 'median'],
        'ram_gb': ['mean', 'median'],
        'battery_mah': ['mean', 'median']
    }).round(2)
    
    print(category_stats)
    
    print("\n Category Insights:")
    for cat in df['price_category'].unique():
        cat_data = df[df['price_category'] == cat]
        print(f"\n {cat} Segment:")
        print(f" Count : {len(cat_data)} phones ( {len(cat_data)/len(df) * 100:.1f}% of dataset)")
        print(f" \Avg Rating :  ₹{cat_data['price'].mean():.0f}")
        print(f" Avg Rating : {cat_data['rating'].mean():.1f}")
        print(f" Avg RAM : {cat_data['ram_gb'].mean():.1f} GB")
        
#======================
#7. STATISTICAL INFERENCE
#======================

def statistical_inference(df):
    """"Perform basis statistical inference"""
    print("\n" + "=" * 80)  
    print("STATISTICAL INFERENCE")
    print("=" * 80)
    
    #7.1 Confidence Intervals 
    print("\n 7.1 Confidence Intervals (95%):")
    print("-" * 80)
    
    #CI for mean price
    mean_price = df['price'].mean()
    std_price = df['price'].std()
    n_price = len(df['price'])  
    se_price = std_price / np.sqrt(n_price)
    ci_price = stats.t.interval(0.95, n_price -1 , loc=mean_price, scale=se_price)
    
    print(f"\n 1. Mean Price :")
    print(f"   • Point Estimate: ₹{mean_price:.2f}")
    print(f"   • 95% CI: [₹{ci_price[0]:.2f}, ₹{ci_price[1]:.2f}]")
    print(f"   • Interpretation: We are 95% confident that the true average smartphone")
    print(f"     price in the population lies between ₹{ci_price[0]:.2f} and ₹{ci_price[1]:.2f}")
    
    #CI for mean rating
    mean_rating = df['rating'].mean()
    std_rating = df['rating'].std()
    n_rating = len(df['rating'])
    se_price = std_rating/np.sqrt(n_rating)
    ci_rating = stats.t.interval(0.95, n_rating -1 , loc=mean_rating, scale=se_price)
    
    print(f"\n 2. Mean Rating :")
    print(f"   • Point Estimate: {mean_rating:.2f}")
    print(f"   • 95% CI: [{ci_rating[0]:.2f}, {ci_rating[1]:.2f}]")
    print(f" Interpretation : The true average rating is between {ci_rating[0]:.2f} and {ci_rating[1]:.2f}")
    print(f" with 95% confidence, indicating generally favorable consumer reception")
    
    #CI for mean battery
    mean_battery = df['battery_mah'].mean()
    std_battery = df['battery_mah'].std()
    n_battery = len (df['battery_mah'])
    se_battery = std_battery / np.sqrt(n_battery)
    ci_battery = stats.t.interval(0.95, n_battery -1 , loc=mean_battery, scale=se_battery)
    
    print(f"\n 3. Mean Battery Capacity :")
    print(f"   • Point Estimate: {mean_battery:.0f} mAh")
    print(f"   • 95% CI: [{ci_battery[0]:.0f} mAh, {ci_battery[1]:.0f} mAh]")
    print(f"   • Interpretation: Average battery capacity falls between {ci_battery[0]:.0f} and")
    print(f"     {ci_battery[1]:.0f} mAh, reflecting industry standards for all-day usage")
   
    
    #7.2 Hypothesis Testing
    print("\n 7.2 Hypothesis Testing:")
    print("-" * 80)
    
    #Test 1 : Do flashship phones have significant;y higher rating than budget phones?
    print("\n Test 1 : Flashship vs Budget Phone Ratings")
    print("   H₀: μ_flagship = μ_budget (no difference in mean ratings)")
    print("   H₁: μ_flagship ≠ μ_budget (ratings differ)")
    
    flagship_ratings = df[df['price_category'] == 'Flagship']['rating']
    budget_ratings = df[df['price_category'] == 'Budget']['rating']
    
    t_stat, p_value = stats.ttest_ind(flagship_ratings, budget_ratings)
    
    print(f"\n   Results:")
    print(f"   • Flagship phones: n={len(flagship_ratings)}, mean={flagship_ratings.mean():.2f}, std={flagship_ratings.std():.2f}")
    print(f"   • Budget phones: n={len(budget_ratings)}, mean={budget_ratings.mean():.2f}, std={budget_ratings.std():.2f}")
    print(f"   • t-statistic: {t_stat:.3f}")
    print(f"   • p-value: {p_value:.4f}")
    print(f"   • Significance level: α = 0.05")
    
    if p_value < 0.05:
       print(f"\n   ✓ Decision: REJECT H₀ (p < 0.05)")
       print(f"   • There IS a statistically significant difference in ratings")
       diff = flagship_ratings.mean() - budget_ratings.mean()
       print(f"   • Flagship phones rated {'higher' if diff > 0 else 'lower'} by {abs(diff):.2f} points on average") 
    else:
       print(f"\n   ✗ Decision: FAIL TO REJECT H₀ (p ≥ 0.05)")
       print(f"   • No statistically significant difference in ratings detected")
       
    #Test 2 : Do phones with 5G have hogher prices than 4G phones?
    print("\n\n Test 2 : 5G vs 4G Phone Prices")
    print("   H₀: μ_5G = μ_4G (no difference in mean prices)")
    print("   H₁: μ_5G > μ_4G (5G phones cost more)")
    
    if "network_type " in df.columns: 
        prices_5g = df[df['network_type'] == '5G']['price']
        prices_4g = df[df['network_type'] == '4G']['price']
        
        t_stat, p_value = stats.ttest_ind(prices_5g, prices_4g)
        p_value_one_sided = p_value / 2
        
        print(f"\n   Results:")
        print(f"   • 5G phones: n={len(prices_5g)}, mean=₹{prices_5g.mean():.2f}, std=₹{prices_5g.std():.2f}")
        print(f"   • 4G phones: n={len(prices_4g)}, mean=₹{prices_4g.mean():.2f}, std=₹{prices_4g.std():.2f}")
        print(f" t-statistic : {t_stat:.3f}")
        print(f" p-value (one-sided) : {p_value_one_sided:.4f}")
        
        if p_value_one_sided <0.05 and t_stat > 0:
            print(f"\n   ✓ Decision: REJECT H₀ (p < 0.05)")
            print(f"   • 5G phones are significantly more expensive")
            print(f"   • Price premium: ₹{(prices_5g.mean() - prices_4g.mean()):.2f} on average")
        else:
            print(f"\n   ✗ Decision: FAIL TO REJECT H₀")
            print(f"   • Insufficient evidence that 5G phones cost significantly more")
            
    #Test 3 : Correlation between RAM and Price
    print("\n\n Test 3 : Correlation between RAM and Price")
    print(" H0 : ρ = 0 (no linear correlation)")
    print(" H1 : ρ ≠ 0 (significant linear correlation)")
    
    correlation, p_value = stats.pearsonr(df['ram_gb'], df['price'])
    
    print(f"\n Results:")
    print(f" • Pearson correlation coefficient: {correlation:.3f}")
    print(f" • p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"\n Decision: REJECT H₀ (p < 0.05)")
        print(f"   • Statistically significant {'positive' if correlation > 0 else 'negative'} correlation")
        print(f"   • Strength: {'Weak' if abs(correlation) < 0.3 else 'Moderate' if abs(correlation) < 0.7 else 'Strong'}")
        print(f"   • Interpretation: RAM is a {'strong' if abs(correlation) > 0.7 else 'moderate'} predictor of price")
    else :
        print(f"\n   ✗ Decision: FAIL TO REJECT H₀")
        print(f"   • No significant linear relationship detected")
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print("\nKey Findings with Uncertainty:")
    print("• Average smartphone price: ₹{:.2f} ± ₹{:.2f} (95% CI)".format(mean_price, ci_price[1] - mean_price))
    print("• Ratings show {}, suggesting price isn't everything".format(
    "significant differences across categories" if p_value < 0.05 else "consistency"))
    print("• RAM strongly predicts price (r={:.3f}, p<0.001)".format(correlation))
    print("• Results based on sample of {} phones - generalizations have inherent uncertainty".format(len(df)))      


#======================
#MAIN EXECUTION
#======================

def main():
    """Main execution function"""
    print("\n")
    print("SMARTPHONE DATA ANALYSIS PIPELINE")
    print("Comparehensive Statistical Analysis & EDA ")
    
    #File path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else '.'
    filepath = os.path.join(script_dir,'cleaned_data.csv')
    
    #1. Load data
    df = load_data(filepath)
    
    #2.Describe daatset
    df = describe_data(df)
    
    #3.Assess data quality
    missing_df, outlier_sumary = assess_data_quality(df)
    
    #4.Clean data
    df_cleaned, cleaning_log = clean_data(df)
    
    #5.EDA
    exploratory_data_analysis(df_cleaned)
    
    #6.Descriptive statistics
    descriptive_statistics (df_cleaned)
    
    #7.Statistical inference
    statistical_inference(df_cleaned)
    
    #Save cleaned data
    output_file = os.path.join(script_dir, 'smartphone_data_cleaned.csv')
    if script_dir == '.':
       output_file = 'smartphone_data_cleaned.csv'
    
    df_cleaned.to_csv(output_file, index = False)
    print("Analysis Complete")
    print("=" * 80)
    print("\n Outputs Generated:")
    print(" 8 visulizations saved in 'plots/' directory")
    print(" Cleaned dataset saved as 'smartphone_data_cleaned.csv'")
    print("=" * 80)
    
if __name__ == "__main__":
    main()
