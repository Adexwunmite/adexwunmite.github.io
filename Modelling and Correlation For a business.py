# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# %%
# import data set
df= pd.read_csv(r"C:\Users\Megalotto DA\Downloads\streamworks_user_data.csv")

# %%
df.info

# %%
df.describe()

# %%
df.value_counts()

# %%
df.isnull().sum()

# %%
#Create a correlation matrix and heatmap
numerical_df = df.select_dtypes(include=[np.number])
print("Numerical columns:", 
numerical_df.columns.tolist())

# %%
#cal the correlation matrix
corr_matrix = numerical_df.corr()

# display the MAT
corr_matrix

# %% [markdown]
# HeatMap

# %%
#Size of plot
plt.Figure(figsize=(10,8))
sns. heatmap(corr_matrix, annot=True, cmap= 'coolwarm', fmt='.2f' , linewidths=0.5)


plt.title('correlation matrix of Numericl variables', fontsize=16 )

plt.show()

# %%
df.head()

# %%
df['signup_date']= pd.to_datetime(df['signup_date'])
df['last_active_date'] =pd.to_datetime(df['last_active_date'])
df.dtypes

# %%
df['tenure_days']= (df['last_active_date']-df['signup_date']).dt.days
df[['signup_date', 'last_active_date', 'tenure_days']].head(10)

# %% [markdown]
# is_loyal = tenure_days > 180 when satisfied Write true for it else 

# %%
df['is_loyal']= df['tenure_days']>180 
df[['tenure_days', 'is_loyal']].head(10)

# %% [markdown]
# Encode categorical features (e.g. LabelEncoder, pd.get_dummies)

# %%
df

# %%
#method1:using label encoder
Label_encoder =LabelEncoder()
df['country_encoded'] = Label_encoder.fit_transform(df['country']) 
df['subscription_encoded']= Label_encoder.fit_transform(df['subscription_type'])
print(df[['country', 'country_encoded', 'subscription_type', 'subscription_encoded']])

# %% [markdown]
# Fill or drop missing values, depending on context

# %%
df.info()
#print('')

# %%
print("missing values count per column:")
print(df.isnull().sum())

# %%
df_clean= df.dropna(subset=['is_churned'])
print(f"missing is_churned value: {df_clean['is_churned'].isnull().sum()}")

# %% [markdown]
# Checking if row was  and dropping gender missing row

# %%


# Verify the missing value has been removed
print(f"Missing values in 'is_churned' after dropping: {df_clean['is_churned'].isnull().sum()}")
print(f"Missing values in 'gender': {df_clean['gender'].isnull().sum()}")
print(f"Original dataset shape: {df.shape}")
print(f"New dataset shape: {df_clean.shape}")

# %%
# Drop rows where either 'signup_date' or 'last_active_date' is missing
df_clean = df_clean.dropna(subset=['signup_date', 'last_active_date','gender'])

# Verify the missing values have been removed
print(f"Missing values in 'signup_date': {df_clean['signup_date'].isnull().sum()}")
print(f"Missing values in 'last_active_date': {df_clean['last_active_date'].isnull().sum()}")
print(f"Dataset shape after dropping date missing values: {df_clean.shape}")

# %%
 
 #Calculate modes for the columns
country_mode = df_clean['country'].mode()[0]  # Get the first mode if multiple exist
subscription_mode = df_clean['subscription_type'].mode()[0]

print(f"Mode for 'country': {country_mode}")
print(f"Mode for 'subscription_type': {subscription_mode}")

# Impute missing values with the mode
df_clean['country'] = df_clean['country'].fillna(country_mode)
df_clean['subscription_type'] = df_clean['subscription_type'].fillna(subscription_mode)

# Verify that missing values have been filled
print(f"Missing values in 'country' after imputation: {df_clean['country'].isnull().sum()}")
print(f"Missing values in 'subscription_type' after imputation: {df_clean['subscription_type'].isnull().sum()}")

# %%
# Impute multiple columns with their modes at once
for column in ['country', 'subscription_type']:
    mode_value = df_clean[column].mode()[0]
    df_clean[column] = df_clean[column].fillna(mode_value)
    print(f"Imputed {column} with mode: {mode_value}")
    print(f"Missing values in {column} after imputation: {df_clean[column].isnull().sum()}")

# %%
# Calculate means for the columns
average_watch_mean = df_clean['average_watch_hours'].mean()
mobile_app_mean = df_clean['mobile_app_usage_pct'].mean()

print(f"Mean for 'average_watch_hours': {average_watch_mean:.2f}")
print(f"Mean for 'mobile_app_usage_pct': {mobile_app_mean:.2f}")

# Impute missing values with the mean
df_clean['average_watch_hours'] = df_clean['average_watch_hours'].fillna(average_watch_mean)
df_clean['mobile_app_usage_pct'] = df_clean['mobile_app_usage_pct'].fillna(mobile_app_mean)

# Verify that missing values have been filled
print(f"Missing values in 'average_watch_hours' after imputation: {df_clean['average_watch_hours'].isnull().sum()}")
print(f"Missing values in 'mobile_app_usage_pct' after imputation: {df_clean['mobile_app_usage_pct'].isnull().sum()}")

# %%
df_clean = df_clean.dropna(subset=['user_id'])
print(f"Missing values in 'user_id' after dropping: {df_clean['user_id'].isnull().sum()}")
print(f"Dataset shape after dropping user_id missing values: {df_clean.shape}")

# For complaints_raised (3 missing) - impute with median
complaints_median = df_clean['complaints_raised'].median()
df_clean['complaints_raised'] = df_clean['complaints_raised'].fillna(complaints_median)
print(f"Imputed 'complaints_raised' with median: {complaints_median}")
print(f"Missing values in 'complaints_raised' after imputation: {df_clean['complaints_raised'].isnull().sum()}")

# For received_promotions (3 missing) - impute with mode since it's categorical
promotions_mode = df_clean['received_promotions'].mode()[0]
df_clean['received_promotions'] = df_clean['received_promotions'].fillna(promotions_mode)
print(f"Imputed 'received_promotions' with mode: {promotions_mode}")
print(f"Missing values in 'received_promotions' after imputation: {df_clean['received_promotions'].isnull().sum()}")

# For referred_by_friend (3 missing) - impute with mode
referral_mode = df_clean['referred_by_friend'].mode()[0]
df_clean['referred_by_friend'] = df_clean['referred_by_friend'].fillna(referral_mode)
print(f"Imputed 'referred_by_friend' with mode: {referral_mode}")
print(f"Missing values in 'referred_by_friend' after imputation: {df_clean['referred_by_friend'].isnull().sum()}")

# For monthly_fee (145 missing) - create flag and impute with median
# First create a flag column to mark which rows had missing monthly_fee
df_clean['monthly_fee_missing'] = df_clean['monthly_fee'].isnull().astype(int)

# %%
# Final check of all missing values
print("\nFinal missing values count for all columns:")
print(df_clean.isnull().sum())

# %%
# For age (3 missing) - impute with median
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)
print(f"Imputed 'age' with median: {age_median}")
print(f"Missing values in 'age' after imputation: {df_clean['age'].isnull().sum()}")

# For monthly_fee (145 missing) - create flag and impute with median
# First create a flag column to mark which rows had missing monthly_fee
df_clean['monthly_fee_missing'] = df_clean['monthly_fee'].isnull().astype(int)

# Then impute with median
fee_median = df_clean['monthly_fee'].median()
df_clean['monthly_fee'] = df_clean['monthly_fee'].fillna(fee_median)
print(f"Imputed 'monthly_fee' with median: {fee_median:.2f}")
print(f"Missing values in 'monthly_fee' after imputation: {df_clean['monthly_fee'].isnull().sum()}")
print(f"Number of rows flagged with missing monthly_fee: {df_clean['monthly_fee_missing'].sum()}")

# Final check of all missing values
print("\nFinal missing values count for all columns:")
print(df_clean.isnull().sum())

# %% [markdown]
# 3. Feature Engineering (Optional)

# %%
#Creating new features:
# 1. tenure_days: Calculate the number of days between signup and last active date
# First ensure both columns are datetime type
df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'])
df_clean['last_active_date'] = pd.to_datetime(df_clean['last_active_date'])

# Calculate tenure in days
df_clean['tenure_days'] = (df_clean['last_active_date'] - df_clean['signup_date']).dt.days

# 2. is_loyal: Create a binary flag for loyal customers
# Define loyal as having tenure above the median
loyal_threshold = df_clean['tenure_days'].median()
df_clean['is_loyal'] = (df_clean['tenure_days'] > loyal_threshold).astype(int)
print(f"Loyalty threshold: {loyal_threshold} days")

# 3. watch_per_fee_ratio: Calculate hours watched per dollar spent
# Handle division by zero by replacing 0 monthly_fee with a small value (0.01)
df_clean['monthly_fee_adj'] = df_clean['monthly_fee'].replace(0, 0.01)
df_clean['watch_per_fee_ratio'] = df_clean['average_watch_hours'] / df_clean['monthly_fee_adj']

# 4. heavy_mobile_user: Create a binary flag for heavy mobile app users
# Define heavy user as above 75th percentile of mobile app usage
mobile_threshold = df_clean['mobile_app_usage_pct'].quantile(0.75)
df_clean['heavy_mobile_user'] = (df_clean['mobile_app_usage_pct'] > mobile_threshold).astype(int)
print(f"Heavy mobile user threshold: {mobile_threshold:.2f}%")

# Display summary of the new features
print("\nSummary of new features:")
print(f"tenure_days - Mean: {df_clean['tenure_days'].mean():.2f}, Median: {df_clean['tenure_days'].median()}")
print(f"is_loyal - Value counts:\n{df_clean['is_loyal'].value_counts()}")
print(f"watch_per_fee_ratio - Mean: {df_clean['watch_per_fee_ratio'].mean():.2f}, Median: {df_clean['watch_per_fee_ratio'].median():.2f}")
print(f"heavy_mobile_user - Value counts:\n{df_clean['heavy_mobile_user'].value_counts()}")

# Drop the temporary column
df_clean = df_clean.drop('monthly_fee_adj', axis=1)


# %%

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# List of numerical columns to examine
numerical_columns = ['user_id','age', 'average_watch_hours', 'mobile_app_usage_pct', 
                     'complaints_raised', 'monthly_fee',
                     'tenure_days', 'watch_per_fee_ratio']

# Create subplots for original distributions
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.ravel()

for i, col in enumerate(numerical_columns):
    if col in df_clean.columns:
        # Plot histogram
        axes[i].hist(df_clean[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')


         # Calculate skewness
        skewness = df_clean[col].skew()
        axes[i].text(0.7, 0.9, f'Skew: {skewness:.2f}', transform=axes[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
    

plt.tight_layout()
plt.show()

# Check skewness for each numerical column
print("Skewness of numerical columns:")
for col in numerical_columns:
    if col in df_clean.columns:
        skew = df_clean[col].skew()
        print(f"{col}: {skew:.4f}")

# Apply transformations where needed
# Rule of thumb: if |skewness| > 1, consider transformation

# For highly skewed positive variables, apply log transformation
highly_skewed_pos = []
for col in numerical_columns:
    if col in df_clean.columns:
        skew = df_clean[col].skew()
        if skew > 1 and df_clean[col].min() > 0:  # Positive skew and all values > 0
            highly_skewed_pos.append(col)
            # Apply log transformation
            df_clean[f'log_{col}'] = np.log1p(df_clean[col])
            print(f"Applied log transformation to {col} (skewness: {skew:.4f})")

# For highly skewed variables that include zero, consider other transformations
highly_skewed = []
for col in numerical_columns:
    if col in df_clean.columns and col not in highly_skewed_pos:
        skew = df_clean[col].skew()
        if abs(skew) > 1:
            highly_skewed.append(col)
            print(f"{col} is highly skewed ({skew:.4f}) but contains zeros or negative values")

# For moderately skewed variables, consider normalization
print("\nNormalizing numerical features...")
scaler = StandardScaler()

# Select columns to normalize (excluding binary flags and already transformed columns)
cols_to_normalize = [col for col in numerical_columns 
                     if col in df_clean.columns 
                     and col not in ['received_promotions', 'referred_by_friend']  # Binary flags
                     and not col.startswith('log_')]  # Already transformed

# Apply standardization
df_clean[cols_to_normalize] = scaler.fit_transform(df_clean[cols_to_normalize])

# Add prefix to normalized columns for clarity
for col in cols_to_normalize:
    df_clean.rename(columns={col: f'std_{col}'}, inplace=True)

# Create visualizations of transformed distributions
transformed_cols = [col for col in df_clean.columns if col.startswith('log_') or col.startswith('std_')]

if transformed_cols:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(transformed_cols[:4]):  # Show first 4 transformed columns
        axes[i].hist(df_clean[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
        # Calculate skewness after transformation
        skewness = df_clean[col].skew()
        axes[i].text(0.7, 0.9, f'Skew: {skewness:.2f}', transform=axes[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Display the skewness after transformations
print("\nSkewness after transformations:")
for col in transformed_cols:
    skew = df_clean[col].skew()
    print(f"{col}: {skew:.4f}")

# Show the first few rows with transformed features
print("\nDataFrame with transformed features:")
print(df_clean[transformed_cols].head())

# %%
df

# %%
from scipy.stats import chi2_contingency
# Assuming your DataFrame is named 'df'
# Let's create a function to perform chi-square tests and interpret results
def chi2_test_and_interpret(df, variable, alpha=0.05):
    # Create contingency table
    contingency_table = pd.crosstab(df['is_churned'], df[variable])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Check if expected frequencies are sufficient (all >=5)
    expected_check = (expected >= 5).all()
    
    # Interpret results
    significance = "significant" if p_value < alpha else "not significant"
    
    # Print results
    print(f"Chi-square test for Churn and {variable}:")
    print("Contingency Table:")
    print(contingency_table)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected frequencies all >=5: {expected_check}")
    
    if p_value < alpha:
        print(f"Conclusion: Significant relationship exists (p < {alpha})")
    else:
        print(f"Conclusion: No significant relationship (p ≥ {alpha})")
    
    # Calculate Cramér's V for effect size (if significant)
    if p_value < alpha:
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * min(contingency_table.shape[0]-1, contingency_table.shape[1]-1)))
        print(f"Cramér's V effect size: {cramers_v:.4f}")
        if cramers_v < 0.1:
            print("Effect size interpretation: Very small")
        elif cramers_v < 0.3:
            print("Effect size interpretation: Small")
        elif cramers_v < 0.5:
            print("Effect size interpretation: Medium")
        else:
            print("Effect size interpretation: Large")
    
    print("\n" + "="*60 + "\n")
    return chi2, p_value

# Perform the tests
print("CHI-SQUARE TEST RESULTS FOR CHURN RELATIONSHIPS\n")
chi2_gender, p_gender = chi2_test_and_interpret(df, 'gender')
chi2_promotions, p_promotions = chi2_test_and_interpret(df, 'received_promotions')
chi2_referral, p_referral = chi2_test_and_interpret(df, 'referred_by_friend')

# Summary table
summary_data = {
    'Variable': ['Gender', 'Received Promotions', 'Referred by Friend'],
    'Chi-square': [chi2_gender, chi2_promotions, chi2_referral],
    'P-value': [p_gender, p_promotions, p_referral],
    'Significant (α=0.05)': [p_gender < 0.05, p_promotions < 0.05, p_referral < 0.05]
}

summary_df = pd.DataFrame(summary_data)
print("SUMMARY OF FINDINGS:")
print(summary_df.to_string(index=False))

# %%

# Separate the data into two groups
churned_watch = df[df['is_churned'] == 1.0]['average_watch_hours']
retained_watch = df[df['is_churned'] == 0.0]['average_watch_hours']

# Check basic statistics
print("DESCRIPTIVE STATISTICS:")
print(f"Churned users (n={len(churned_watch)}):")
print(f"  Mean: {churned_watch.mean():.2f} hours")
print(f"  Std: {churned_watch.std():.2f}")
print(f"  Min: {churned_watch.min():.2f}, Max: {churned_watch.max():.2f}")

print(f"\nRetained users (n={len(retained_watch)}):")
print(f"  Mean: {retained_watch.mean():.2f} hours")
print(f"  Std: {retained_watch.std():.2f}")
print(f"  Min: {retained_watch.min():.2f}, Max: {retained_watch.max():.2f}")

# Check assumptions for t-test
# 1. Normality (using Shapiro-Wilk test)
_, p_churned_norm = stats.shapiro(churned_watch)
_, p_retained_norm = stats.shapiro(retained_watch)

print(f"\nNORMALITY TEST (Shapiro-Wilk):")
print(f"Churned users p-value: {p_churned_norm:.4f}")
print(f"Retained users p-value: {p_retained_norm:.4f}")

# 2. Homogeneity of variances (Levene's test)
_, p_levene = stats.levene(churned_watch, retained_watch)
print(f"\nEQUAL VARIANCE TEST (Levene's): p-value = {p_levene:.4f}")

# Perform appropriate t-test based on variance equality
if p_levene < 0.05:
    # Unequal variances, use Welch's t-test
    t_stat, p_value = stats.ttest_ind(churned_watch, retained_watch, equal_var=False)
    test_type = "Welch's t-test (unequal variances)"
else:
    # Equal variances, use standard t-test
    t_stat, p_value = stats.ttest_ind(churned_watch, retained_watch, equal_var=True)
    test_type = "Standard t-test (equal variances)"

print(f"\nT-TEST RESULTS ({test_type}):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt(((len(churned_watch)-1)*churned_watch.var() + (len(retained_watch)-1)*retained_watch.var()) / 
                     (len(churned_watch) + len(retained_watch) - 2))
cohens_d = (churned_watch.mean() - retained_watch.mean()) / pooled_std
print(f"Cohen's d effect size: {cohens_d:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"\nCONCLUSION: There is a significant difference in watch time between churned and retained users (p < {alpha}).")
    if cohens_d < 0:
        print("Churned users have significantly lower watch time than retained users.")
    else:
        print("Churned users have significantly higher watch time than retained users.")
else:
    print(f"\nCONCLUSION: There is no significant difference in watch time between churned and retained users (p ≥ {alpha}).")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_size = "very small"
elif abs(cohens_d) < 0.5:
    effect_size = "small"
elif abs(cohens_d) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"The effect size is {effect_size} (|d| = {abs(cohens_d):.3f}).")

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_churned', y='average_watch_hours', data=df)
plt.title('Average Watch Hours by Churn Status')
plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')
plt.ylabel('Average Watch Hours')
plt.show()

# Additional visualization: Kernel Density Estimate
plt.figure(figsize=(10, 6))
sns.kdeplot(churned_watch, label='Churned Users', fill=True)
sns.kdeplot(retained_watch, label='Retained Users', fill=True)
plt.title('Distribution of Watch Hours by Churn Status')
plt.xlabel('Average Watch Hours')
plt.ylabel('Density')
plt.legend()
plt.show()

# %%


# Select numerical variables for correlation analysis
numerical_vars = ['age', 'average_watch_hours', 'mobile_app_usage_pct', 
                  'complaints_raised', 'monthly_fee', 'tenure_days', 
                  'country_encoded', 'subscription_encoded', 'is_churned']

# Create correlation matrix
corr_matrix = df[numerical_vars].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
            annot=True, fmt=".2f")

plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()

# Focus on correlations with churn
churn_correlations = corr_matrix['is_churned'].sort_values(ascending=False)

print("CORRELATIONS WITH CHURN (is_churned):")
print("=" * 40)
for idx, value in churn_correlations.items():
    if idx != 'is_churned':  # Skip correlation with itself
        print(f"{idx}: {value:.3f}")

# Identify significant correlations (|r| > 0.1)
significant_correlations = churn_correlations[(abs(churn_correlations) > 0.1) & (churn_correlations.index != 'is_churned')]
print("\nSIGNIFICANT CORRELATIONS (|r| > 0.1):")
print("=" * 40)
for idx, value in significant_correlations.items():
    print(f"{idx}: {value:.3f}")

# Calculate point-biserial correlations for categorical variables with churn
categorical_vars = ['gender', 'received_promotions', 'referred_by_friend', 'is_loyal']

print("\nPOINT-BISERIAL CORRELATIONS FOR CATEGORICAL VARIABLES:")
print("=" * 50)

for var in categorical_vars:
    # Convert categorical to numerical
    if var == 'gender':
        # Create dummy variables
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
        for col in gender_dummies.columns:
            r, p_value = stats.pointbiserialr(gender_dummies[col], df['is_churned'])
            print(f"{col}: r = {r:.3f}, p = {p_value:.4f}")
    else:
        # Convert yes/no to 1/0
        if var in ['received_promotions', 'referred_by_friend']:
            mapped_var = df[var].map({'Yes': 1, 'No': 0})
        else:  # is_loyal
            mapped_var = df[var].astype(int)
        
        r, p_value = stats.pointbiserialr(mapped_var, df['is_churned'])
        print(f"{var}: r = {r:.3f}, p = {p_value:.4f}")

# Create scatter plots for variables with strongest correlations to churn
strong_correlations = churn_correlations.abs().sort_values(ascending=False)
top_3_vars = strong_correlations.index[1:4]  # Skip churn itself

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, var in enumerate(top_3_vars):
    sns.scatterplot(x=df[var], y=df['is_churned'], alpha=0.5, ax=axes[i])
    axes[i].set_title(f'{var} vs. Churn\nr = {churn_correlations[var]:.3f}')
    axes[i].set_ylabel('Churn (0=No, 1=Yes)')

plt.tight_layout()
plt.show()



# %%
# Set the visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Assuming your DataFrame is named 'df'

# 1. Create a summary of churn rate
churn_rate = df['is_churned'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
bars = plt.bar(['Active', 'Churned'], churn_rate, color=['steelblue', 'lightcoral'])
plt.title('Overall Churn Rate')
plt.ylabel('Percentage of Users')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{churn_rate[i]:.1f}%', ha='center')
plt.show()


# %%
# 2. Numerical variables comparison
numerical_vars = ['age', 'average_watch_hours', 'mobile_app_usage_pct', 
                  'complaints_raised', 'monthly_fee', 'tenure_days']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, var in enumerate(numerical_vars):
    # Create boxplot
    sns.boxplot(x='is_churned', y=var, data=df, ax=axes[i])
    axes[i].set_title(f'{var} by Churn Status')
    axes[i].set_xlabel('Churn Status (0=Active, 1=Churned)')
    axes[i].set_ylabel(var)
    
    # Calculate and display t-test results
    active_data = df[df['is_churned'] == 0][var]
    churned_data = df[df['is_churned'] == 1][var]
    t_stat, p_value = stats.ttest_ind(active_data, churned_data, nan_policy='omit')
    
    # Add significance annotation
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
        
    axes[i].text(0.5, 0.95, f't-test: p={p_value:.4f} {sig_text}', 
                transform=axes[i].transAxes, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# 3. Categorical variables comparison
categorical_vars = ['gender', 'subscription_type', 'received_promotions', 
                    'referred_by_friend', 'is_loyal', 'country']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, var in enumerate(categorical_vars):
    # Calculate percentages
    crosstab = pd.crosstab(df[var], df['is_churned'], normalize='index') * 100
    
    # Plot
    crosstab.plot(kind='bar', ax=axes[i], color=['steelblue', 'lightcoral'])
    axes[i].set_title(f'Churn Rate by {var}')
    axes[i].set_ylabel('Percentage')
    axes[i].legend(['Active', 'Churned'])
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add chi-square test results
    contingency_table = pd.crosstab(df[var], df['is_churned'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # Add significance annotation
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
        
    axes[i].text(0.5, 0.95, f'χ²: p={p_value:.4f} {sig_text}', 
                transform=axes[i].transAxes, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# 4. Distribution comparison for key variables
key_vars = ['average_watch_hours', 'tenure_days', 'complaints_raised']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, var in enumerate(key_vars):
    # Create histogram with density curve
    sns.histplot(data=df, x=var, hue='is_churned', kde=True, 
                 element='step', stat='density', common_norm=False, ax=axes[i])
    axes[i].set_title(f'Distribution of {var} by Churn Status')
    axes[i].set_xlabel(var)

plt.tight_layout()
plt.show()


# %%
df

# %%
df = df.dropna(subset=['is_churned'])

# %%
string_mask= df['is_churned'].apply(lambda x: isinstance (x,str))
has_strings= string_mask.any()
print(f"Dors is_churned has string value? (has string)")

if hss_string:

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer



# Encode binary categorical variables
df['received_promotions'] = df['received_promotions'].map({'Yes': 1, 'No': 0})
df['referred_by_friend'] = df['referred_by_friend'].map({'Yes': 1, 'No': 0})
#df['gender']=df['gender'].map({''})


# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing target values
df = df.dropna(subset=['is_churned'])

# Separate features and target
X = df.drop('is_churned', axis=1)
y = df['is_churned']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#model.fit(X_train, y_train)
# Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train,y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# %%
# Check current number of rows
print(f"Original dataset shape: {df.shape}")

# Remove rows where is_churned is NaN
df_clean = df.dropna(subset=['is_churned'])

# Check new number of rows
print(f"Dataset shape after removing NaN from is_churned: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} rows")

# Verify no NaN values remain in is_churned
print(f"NaN values in is_churned after cleaning: {df_clean['is_churned'].isnull().sum()}")

# %% [markdown]
# 

# %%
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
# Drop irrelevant columns
    columns_to_drop = ['user_id', 'signup_date', 'last_active_date', 'country', 
                       'subscription_type', 'is_loyal']
    df_processed = df_processed.drop(columns_to_drop, axis=1, errors='ignore')

    # Encode binary categorical variables
    df_processed['received_promotions'] = df_processed['received_promotions'].map({'Yes': 1, 'No': 0})
    df_processed['referred_by_friend'] = df_processed['referred_by_friend'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode gender (if it exists)
    if 'gender' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['gender'], drop_first=True)
    
    # Check for missing values
    print("Missing values in each column:")
    print(df_processed.isnull().sum())
    
    # Drop rows with missing target values
    df_processed = df_processed.dropna(subset=['is_churned'])
    
    return df_processed

# Preprocess the data
df_processed = preprocess_data(df)

# Separate features and target
X = df_processed.drop('is_churned', axis=1)
y = df_processed['is_churned']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for better visualization (optional)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Scaled training data sample:")
print(X_train_scaled_df.head())

# Train the logistic regression model
model = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance if present
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.coef_[0]
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

# Display model coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


