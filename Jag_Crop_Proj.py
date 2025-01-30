#  Crop EDA analysis developed by Jagadeesh V
#  Created by Jagadeesh V during Second week of Jan. 2025

# Step 1: Understanding the Data
# Step 2: Pre-Processing / Cleaning Data 
#          A. Duplicates, B. Missing Data, C. Data type conversion, D. Outliers
# Step 3: Plot Chart
# Step 4: Streamlit Application
#          A. Merging Data & re-treating Outliers
#          B. Transforming Data
#          C. ML - Training data. Random Forest Regressor
#          D. 
# Step 5: Getting user inputs & Prediction
# Step 6: EDA Report
##### ----------------------------------------------------------------------------------- #####
# 0. Importing required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.text import Text
import os

console = Console()
plt.style.use("ggplot")
##### ----------------------------------------------------------------------------------- #####

BOLD = '\033[1m'
REDWHITE = '\033[151;75;41m'
NAVYBLUE = '\033[38;5;93m'
BROWN = '\033[38;5;94m'
BLUE = '\033[36;75;94m'
RESET = '\033[0m'

##### Main Routine  #####
# M0. Heading display & uncleaned data details
print('--------------------------------------------------------------------------------')
print('')
console.print("[bold red]Predicting Crop Production Based on Agricultural Data\n[/]", justify="center")
print(f"{BROWN}Steps{RESET}")
print(f"{BROWN}1. Raw data Preparing, 2. Analysis & Charts, 3. ML, 4. Prediction, 5. EDA Report{RESET} \n")

raw_df = pd.read_csv("D:\Guvi\Crop Proj\FAOSTAT_data\FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv")
file = "FAOSTAT"

console.print("[purple]1. Raw data Cleansing of "+ file+ " Dataset\n", justify='center')
print(f"{BOLD}{NAVYBLUE}A. The intial uncleaned data Shape, Desciption & Structure\n{RESET}")
raw_df.head()
print('')
print(f"Data Shape: {raw_df.shape}")
print()
raw_df.describe()
print()
print (raw_df.info())
print()
input("Press ENTER key to continue...")
print()
raw_df.drop('Year Code', axis=1, inplace=True)

#    M1. Data Cleansing & Preparation
##  M1a. Check and clear Duplicate Values
try:
    print(f"{BOLD}{NAVYBLUE}B. Checking and Clearing Duplicates in {file}\n{RESET}")
    if  raw_df.duplicated().sum() > 0:
        print(f'Duplicates: \n  {df_cust.duplicated().sum()} \n')
        raw_df.drop_duplicates(inplace=True)
        print(f'Duplicates in {file} Cleared \n')
    else:
        print(f'No Duplicates in {file} \n')
except Exception as e:
    print(f'Error in duplicate check function  for {file} - {e}')
input("Press ENTER key to continue...")
print()

##  M1b  Check and clear Null / Missing values
## 1. Using Mean & Mode for numeric & Other types,    2. Forward fill  &   3. backward fill
try:
    print(f"{BOLD}{NAVYBLUE}C. Checking and Clearing Null Values in {file}\n{RESET}")
    raw_df.replace(0, np.nan, inplace=True)
    null_col = raw_df.isnull().sum()[raw_df.isnull().sum() > 0]
    if not null_col.empty:
        print(f"Null / missing values Found in columns: \n {null_col} \n")
        print(f"Updating the same with Mean for numeric and Mode for Non numeric\n")
        for column in raw_df.columns:
            if raw_df[column].dtype in ['int64', 'float64']:
                raw_df[column] = raw_df[column].fillna(raw_df[column].mean())
#                raw_df[column].fillna(raw_df[column].mean(), inplace=True)
#                mean_value = raw_df[column].mean()
#                raw_df[column].fillna(mean_value, inplace=True)
            else:
                raw_df[column] = raw_df[column].fillna(raw_df[column].mode()[0])
#                raw_df[column].fillna(raw_df[column].mode()[0], inplace=True)
        null_col = raw_df.isnull().sum()[raw_df.isnull().sum() > 0]
        if not null_col.empty:
            print(f"Further null values clearing thru' forward filling for columns: \n {null_col} \n")
            raw_df.ffill(inplace=True)
            # Checking for 1st Row missing values to use bfill
            null_cols = raw_df.iloc[0].isnull()
            null_cols_list = null_cols[null_cols].index.tolist()
            if null_cols_list:
                print("Clearing also with backward fill as Null value are in first rows of \n :\n", null_cols_list)
                print('')
                raw_df.bfill(inplace=True)
    else:
        print("No Null values found\n")
    ###  Confirming No missing values
    null_col = raw_df.isnull().sum()[raw_df.isnull().sum() > 0]
    print(f"Null values result :: Missing values:\n", null_col, "\n")
except Exception as e:
    print('Error in Check & clear null values function')
input("Press ENTER key to continue...")
print()
print(f"{BOLD}{NAVYBLUE}D. Other cleansing in {file}\n{RESET}")
print('No Objects with Date, Currency, Zip codes, etc to clean\n')
input("Press ENTER key to continue...")
print()

##### ---------------------------------------------------------------------------- #####

console.print("[purple]2.  Analysis  &  Charts\n", justify='center')
print("\n")
# M2 Ploting graphs by filtering required columns for Analysis
## M2a. Filtering required Columns 
relv_col = ['Area', 'Item', 'Element', 'Year', 'Value']
relevant_df = raw_df[relv_col]
print(f"{BOLD}{NAVYBLUE}A. Cleansed & relevant filtered Data \n{RESET}")
print(relevant_df)
relevant_df.to_csv('D:\Guvi\Crop Proj\FAOSTAT_data\FAOSTAT_Cleaned_Req.csv', index=False)
print('Created CSV file -> D:\Guvi\Crop Proj\FAOSTAT_data\FAOSTAT_Cleaned_Req.csv \n')
input("Press ENTER key to continue...")
print()

## M2a. Data analysis & Graphical Representation ##
## M2a1. Most & Least cultivated Items across regions
cd = relevant_df['Item'].value_counts()
most_cultivated = cd.idxmax()
least_cultivated = cd.idxmin() 

print(f"{BOLD}{NAVYBLUE}B. Most & Least Cultivated Crops \n{RESET}")
print("Most Cultivated Crops: ", most_cultivated)
print("Least Cultivated Crops: ", least_cultivated,"\n") 
input("Press ENTER key to continue...")
print()

print(f"{BOLD}{NAVYBLUE}C. Graphical Representations \n{RESET}")
## M2a2.  Top 10 Cultivated Crops across Regions and years
plt.figure(figsize=(13,6))
cd.head(10).plot(kind='bar', color='plum')
plt.title("Top 10 Cultivated Crops across Regions and years") 
plt.xlabel("Crop Item")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## M2a3.  Regionwise Cultivation trend
regions = relevant_df['Area'].value_counts()
most_active = regions.idxmax()
least_active = regions.idxmin()

plt.figure(figsize=(13,6))
regions.head(10).plot(kind='bar', color = 'Turquoise')
plt.title("Top 10 regions with highest Agricultural activity")
plt.xlabel("Region")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Region/ Item wise Cultivation trend [ Area, Crop-wise Higher Activity ]
grouped_df = relevant_df.groupby(['Area', 'Item'])['Value'].sum().reset_index()
top_10 = grouped_df.sort_values(by='Value', ascending=False).head(15)
pivot_df = top_10.pivot(index='Area', columns='Item', values='Value')

pivot_df.plot(kind='bar', figsize=(12, 6))
plt.title('Area, Crop-wise Higher Activity')
plt.xlabel('Area')
plt.ylabel('Value')
plt.legend(title='Item')
plt.xticks(rotation=0)
plt.show()

area_df = relevant_df[relevant_df['Element'] == 'Area harvested'] 
area_df['Value'] = area_df['Value']/10000000
area_trends = area_df.groupby('Year')['Value'].sum() #.reset_index()

yield_df = relevant_df[relevant_df['Element'] == 'Yield'] 
yield_trends = yield_df.groupby('Year')['Value'].mean()

production_df = relevant_df[relevant_df['Element'] == 'Production']
production_trends = production_df.groupby('Year')['Value'].sum() 

#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()
plt.figure(figsize=(13,6)) 
plt.subplot(1, 3, 1)
area_trends.plot(color='blue', marker='o')
plt.title("Yearly Trend: Area Harvested")
plt.xlabel("Year")
plt.ylabel("Total Area (ha)")
plt.grid()

plt.subplot(1, 3, 2)
yield_trends.plot(color='blue', marker='o') 
plt.title("Yearly Trend: Yield")
plt.xlabel("Year")
plt.ylabel("Yield (kg/ha)")
plt.grid()

plt.subplot(1, 3, 3) 
production_trends.plot(color='blue', marker='o') 
plt.title("Yearly Trend: Production")
plt.xlabel("Year")
plt.ylabel("Total Production(Tons)")
plt.grid()

plt.tight_layout()
plt.show() 

crop_production_trends = production_df.groupby(['Year', 'Item'])['Value'].sum().unstack()
region_production_trends = production_df.groupby(['Year', 'Area'])['Value'].sum().unstack()

crop_growth_trends = crop_production_trends.diff().mean().sort_values(ascending=False)
print(f"{BOLD}{NAVYBLUE}C. Increasing & Decreasing Trends \n{RESET}")
print("Crops with Increasing Trends:\n", crop_growth_trends[crop_growth_trends > 0].head())
print("\nCrops with Decreasing Trends:\n", crop_growth_trends[crop_growth_trends < 0].head())

region_growth_trends = region_production_trends.diff().mean().sort_values(ascending=False)
print("Regions with Increasing Trends:\n", region_growth_trends[region_growth_trends > 0].head())
print("\nRegions with Decreasing Trends:\n", region_growth_trends[region_growth_trends < 0].head())
print()
input("Press ENTER key to continue...")
print()

plt.figure(figsize=(13,6))
##### ------------------------------------------------------------------------------- #####

### Outliers Using IQR (Interquartile Range)
def find_iqr_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75) 
    IQR = Q3 - Q1 
    lower_limit = Q1 - 1.5 * IQR 
    upper_limit = Q3 + 1.5 * IQR 
    return (column < lower_limit) | (column > upper_limit) 

#relevant_df[conc] = relevant_df['Area']+ relevant_df['Item']+ relevant_df['Element']+ relevant_df['Year']

yield_df = relevant_df[relevant_df['Element'] == 'Yield'][['Item', 'Area', 'Year', 'Value']]
yield_df.rename(columns={'Value': 'Yield'}, inplace=True)

production_df = relevant_df[relevant_df['Element'] == 'Production'][['Item', 'Area', 'Year', 'Value']]
production_df.rename(columns={'Value': 'Production'}, inplace=True)

prod_yld_df = pd.merge(yield_df, production_df, on=['Item', 'Area', 'Year'], suffixes=('_Yield', '_Production'))
print(f"{BOLD}{NAVYBLUE}D. O u t  l i e r s \n{RESET}")
print("###########################################################################################")
print(prod_yld_df)
print("###########################################################################################")
prod_yld_df['Yield_Outliers'] = find_iqr_outliers(prod_yld_df['Yield'])
prod_yld_df['Production_Outliers'] = find_iqr_outliers(prod_yld_df['Production'])

yield_outliers = prod_yld_df[prod_yld_df['Yield_Outliers']]
production_outliers = prod_yld_df[prod_yld_df['Production_Outliers']]

print("Yield Outliers:")
print(yield_outliers)

print("\nProduction Outliers:")
print(production_outliers)
print()
input("Press ENTER key to continue...")
print()

plt.figure(figsize=(12, 6))
#top_15_df = prod_yld_df.nlargest(15, 'Yield')
#sns.boxplot(data = top_15_df, x='Item', y='Yield', showfliers=True)
sns.boxplot(data = prod_yld_df, x='Item', y='Yield', showfliers=True)
plt.xticks(rotation=45, ha='right')
plt.title("Boxplot of Production by Crop (With Outliers)")                          
plt.ylabel("Production (tons)")
plt.tight_layout()
plt.show()

yield_outliers_by_year = yield_outliers.groupby('Year')['Yield'].count()
production_outliers_by_year = production_outliers.groupby('Year')['Production'].count()

plt.figure(figsize=(12, 6))
yield_outliers_by_year.plot(kind='bar', color='red', alpha=0.7, label='Yield Outliers')                          
production_outliers_by_year.plot(kind='bar', color='blue', alpha=0.7, label='Production Outliers')
plt.title("Outliers Detected Over Time")                            
plt.xlabel("Year")                             
plt.ylabel("Number of Outliers")                           
plt.legend()
plt.tight_layout()
plt.show()

yield_outliers_by_area = yield_outliers['Area'].value_counts()
production_outliers_by_area = production_outliers['Area'].value_counts()

print("\nRegions with Yield Outliers:")                           
print(yield_outliers_by_area)

print("\nRegions with Prooduction Outliers:")                             
print(production_outliers_by_area)
print()
input("Press ENTER key to continue...")
print()

############################## Outliers removed from relevant dataset
############################## df_no_outliers = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

##### ------------------------------------------------------------------------------------------------------  #####
### Machine Learning 
### Categorical columns converting to Numeric
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
console.print("[purple]3. Machine Learning\n", justify='center')
categorical_columns = relevant_df.select_dtypes(include=['object', 'category']).columns
print(f"{BOLD}{NAVYBLUE}A. Categorical columns conversion - One Hot Coding\n{RESET}")
print("Categorical columns:", categorical_columns)

### One Hot Coding
for col in categorical_columns:
    relevant_df[col] = label_encoder.fit_transform(relevant_df[col])
relevant_df.to_csv('D:\Guvi\Crop Proj\FAOSTAT_data\FAOSTAT_Cleaned1.csv', index=False)

##### ------------------------------------------------------------------------------------------------------  #####
#### Linear Regression

##import pandas as pd
##from sklearn.model_selection import train_test_split
##from sklearn.linear_model import LinearRegression
##from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
##from sklearn.ensemble import RandomForestRegressor
##
##X = relevant_df[['Area', 'Item','Element', 'Year']]  # Independent variables
##y = relevant_df['Value']  # Dependent variable
##
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##
##linear_model = LinearRegression()
##linear_model.fit(X_train, y_train)
##
##y_pred = linear_model.predict(X_test)
##
##mse = mean_squared_error(y_test, y_pred)
##mae = mean_absolute_error(y_test, y_pred)
##r2 = r2_score(y_test, y_pred)
##
##print(f"{BOLD}{NAVYBLUE}\nB. Linear Regression Model Evaluation\n{RESET}")
##print(f"Mean Squared Error (MSE): {mse}")
##print(f"Mean Absolute Error (MAE): {mae}")
##print(f"R-squared Score (R2): {r2} \n")
##
###### Random, Forest Regression
##X = relevant_df[['Area', 'Item', 'Element', 'Year']]
##y = relevant_df['Value']
##
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
##
##rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 
##rf_model.fit(X_train, y_train) 
##
##y_pred = rf_model.predict(X_test) 
##
##print(f"{BOLD}{NAVYBLUE}C. Random Forest Regression Model Evaluation\n{RESET}")
##print("MSE", mean_squared_error(y_test, y_pred))
##print("R2 Score:", r2_score(y_test, y_pred))
##
###### Gradient Boosting Regression
##from sklearn.ensemble import GradientBoostingRegressor
##
##gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
##
##gb_model.fit(X_train, y_train) 
##
##y_pred = gb_model.predict(X_test) 
##
##print(f"{BOLD}{NAVYBLUE}\nD. Gradient Boosting Regression Model Evaluation\n{RESET}")
##print("MSE:", mean_squared_error(y_test, y_pred))
##print("R2 Score:", r2_score(y_test, y_pred))

console.print("[purple]4. Prediction\n", justify='center')
console.print(" \n", justify='center')
input("Press ENTER key")
os.system('streamlit run Jag_Crop_proj_Predict_app.py "D:\Guvi\Crop Proj\FAOSTAT_data\FAOSTAT_Cleaned_Req.csv"')
print('')
