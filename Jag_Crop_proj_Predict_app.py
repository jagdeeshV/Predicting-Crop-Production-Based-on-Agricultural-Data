# Streamlit application for Crop Production Prediction
# 0. Required packages importing
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import time
import keyboard
import os
import psutil
import sys
## --------------------------------------------------------------------------------------------------------------------------------------- ##

# 1. The Merging, Outliers, feature transformation, Training, etc functions called by Main Streamlit routin
class CropProductionClass:
# 1A. Class variable initialization 
    def __init__(self):
        self.num_area = LabelEncoder()
        self.num_item = LabelEncoder()
        self.encoded_names = ['encoded_area', 'encoded_item', 'Year', 'Area_harvested', 'Yield']
        
        # Create preprocessing pipeline
        numeric_features = ['Year', 'Area_harvested', 'Yield']
        categorical_features = ['encoded_area', 'encoded_item']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())  # RobustScaler handles outliers better
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Create model pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,  # Increased number of trees
                max_depth=20,      # Limited depth to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            ))
        ])

# 1B. Merging  & treating outliers rows based on Element (Area harvested, Production & yield
    def Merge_Element_data(self, csv_file):
    # 1Ba. Merging  & treating outliers rows based on Element (Area harvested, Production & yield
        try:
            df = pd.read_csv(csv_file)
            # cols are 'Area', 'Item', 'Element', 'Year', 'Value'
            # Make values of Area_harvested, Yield & Production in single rows for easy and better analysis & prediction 
            # Create dataframes for each element
            area_harvested_df = df[df['Element'] == 'Area harvested'].copy()
            yield_df = df[df['Element'] == 'Yield'].copy()
            production_df = df[df['Element'] == 'Production'].copy()
            
            # Merge the dataframes
            merged_df = pd.merge(
                area_harvested_df[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Area_harvested'}),
                yield_df[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Yield'}),
                on=['Area', 'Item', 'Year'],
                how='inner'
            )
            
            merged_df = pd.merge(
                merged_df,
                production_df[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Production'}),
                on=['Area', 'Item', 'Year'],
                how='inner'
            )
            
    # 1Bb. Removing outliers using IQR method
            def remove_outliers(df, column):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            # Apply outlier removal to numerical columns
            merged_df = remove_outliers(merged_df, 'Area_harvested')
            merged_df = remove_outliers(merged_df, 'Yield')
            merged_df = remove_outliers(merged_df, 'Production')
            
            # Log transform production values
            merged_df['Production_log'] = np.log1p(merged_df['Production'])
            
            # Remove rows with missing values
            merged_df = merged_df.dropna()
            
            if merged_df.empty:
                raise ValueError("No data remaining after preprocessing")
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error in data loading and cleaning: {str(e)}")
            return None
    
# 1C. Preparing fatures (Categorical columns to Numeric transformation)
    def prepare_features(self, df):
        try:
            if df is None or df.empty:
                raise ValueError("Empty dataframe provided to prepare_features")
            
            # Encode categorical variables
            df['encoded_area'] = self.num_area.fit_transform(df['Area'])
            df['encoded_item'] = self.num_item.fit_transform(df['Item'])
            
            # Create feature matrix as DataFrame with named columns
            x = pd.DataFrame(df[self.encoded_names])
            y = df['Production_log']  # Use log-transformed target
            
            return x, y
            
        except Exception as e:
            st.error(f"Error in feature preparation: {str(e)}")
            print(f"Error in feature preparation: {str(e)}")
            return None, None
    
# 1D. Training Model, Prediction and calculating R2, MSE, MAE & CV score metrics 
    def train_model(self, x, y):
        try:
            if x is None or y is None:
                raise ValueError("Features or target is None")
            
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(x_train, y_train)
            
            # Prediction log
            y_pred_log = self.model.predict(x_test)
            
            # Transform predictions to original scale
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test)
            
            # Calculate metrics on original scale
            metrics = {
                'R2': r2_score(y_test_original, y_pred),
                'MSE': mean_squared_error(y_test_original, y_pred),
                'MAE': mean_absolute_error(y_test_original, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred)),
                'CV_scores': cross_val_score(self.model, x, y, cv=5)
            }
            
            # Calculate percentage errors
            metrics['MAPE'] = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
            
            return metrics
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            print(f"Error in model training: {str(e)}")
            return None
##### --------------------------------------------------------------------------------------------------------------------------------------- #####

# 2. Main Streamklit routine Accepting user inputs, calling functions & Predicting
    def main_ui(self, csv_file):
        try:
            st.title('Crop Production - Prediction')
            st.write(f"\nCleaned Data set Loaded : {csv_file}")
            temp_msg = st.empty()
            temp_msg.text("\n Wait. Merging, preparing features and training the dataset")
# 2A. Accepting the Cleaned data set
            #csv_file = st.file_uploader("Upload Cleaned FAOSTAT  CSV file", type=['csv'])
            
            if 'one_time' not in st.session_state:
                st.session_state.one_time = 0
            if 'first_time' not in st.session_state:
                st.session_state.first_time = 0

            if csv_file is not None:
# 2B. Mergin based on element
                df = self.Merge_Element_data(csv_file)
                #print(f" Merge {datetime.now().time()}")
                temp_msg.text("\n Wait. Merging, preparing features and training the dataset ...")

                if df is not None and not df.empty:
# 2C. Categorical to Numric transformation
                    x, y = self.prepare_features(df)
                    temp_msg.text("\n Wait. Merging, preparing features and training the dataset ......")
                    
                    if x is not None and y is not None:
# 2D. First time Model training inpuuting CSV file, mergin,  & transforming
#        subsequently executed on click of Predict button [after user inputs ] to avoid processing time on each and every input
                        if st.session_state.one_time == 0:
                            st.session_state.one_time = 1
                            metrics = self.train_model(x, y)
                            temp_msg.text("\n Wait. Merging, preparing features and training the dataset .........")
                            
                            if metrics is not None:
                                html_code = """
                                <div style="text-align: center; font-size: 24px;">
                                 Model Performance of cleaned and pre-processed full dataset
                                </div>
                                """
                                st.markdown(html_code, unsafe_allow_html=True)
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                                with col2:
                                    st.metric("Root Mean Sq. Error", f"{metrics['RMSE']:.2f}")
                                with col3:
                                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")

                        df.to_csv('D:\Guvi\Crop Proj\FAOSTAT_data\Final FAOSTAT_Processed.csv', index=False)
# 2Db. Input parameters for prediction
                        st.markdown(
                            """
                            <style>
                            .css-1d391kg {  # This class name might change with Streamlit updates
                                width: 300px; 
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        st.sidebar.header('Input Parameters')
                        temp_msg.empty()
                        st.markdown("""
                            <style>
                            .stSelectbox label {
                                display: inline-block;
                                margin-right: 9px;
                            }
                            .stSelectbox div[data-baseweb="select"] {
                                display: inline-block;
                                width: auto;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                        selected_area = st.sidebar.selectbox('Area', self.num_area.classes_)
                        st.markdown("""
                            <style>
                            .stSelectbox label {
                                display: inline-block;
                                margin-right: 9px;
                            }
                            .stSelectbox div[data-baseweb="select"] {
                                display: inline-block;
                                width: auto;
                            }
                            </style>
                            """, unsafe_allow_html=True)
#                        selected_item = st.sidebar.selectbox('Crop', self.num_item.classes_)
                        selected_item = st.sidebar.selectbox('Crop', df[df['Area'] == selected_area]['Item'].unique())
                        min_yr = df['Year'].min()
                        max_yr = df['Year'].max()
                        future_yr = round(max_yr, -1)
                        future_predict = st.sidebar.checkbox('Predict Future(2024-2030)? ')
                        if future_yr <= max_yr:
                            if future_predict:
                                future_yr = future_yr+ 10
                            else:
                                future_yr = max_yr
                        selected_year = st.sidebar.slider('Select Year', min_yr, future_yr, max_yr)
                    # Use median values from similar crops/areas as default
                        default_area = df[(df['Item'] == selected_item) & (df['Area'] == selected_area)]['Area_harvested'].median()
                        # default_area = df[df['Item'] == selected_item]['Area_harvested'].median()
                        default_yield = df[df['Item'] == selected_item]['Yield'].median()
                        area_harvested = st.sidebar.number_input('Area_harvested (ha)', 
                                                               min_value=0.0, 
                                                               value=float(default_area))
                        yield_value = st.sidebar.number_input('Yield (kg/ha)', 
                                                            min_value=0.0,
                                                            value=float(default_yield))

                        if st.sidebar.button('Predict'):
# 2E. Prediction based on inputs
                            if st.session_state.one_time == 1:
# 2E1 Subsequent times Model training o clicking predct button to avoid processing time on each and every input
                               metrics = self.train_model(x, y)

                               if metrics is not None:
                                    html_code = """
                                    <div style="text-align: center; font-size: 24px;">
                                     Model Performance
                                    </div>
                                    """
                                    st.markdown(html_code, unsafe_allow_html=True)
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("R² Score", f"{metrics['R2']:.3f}")
                                    with col2:
                                        st.metric("Root Mean Sq. Error", f"{metrics['RMSE']:.2f}")
                                    with col3:
                                        st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
                               
                            input_data = pd.DataFrame([[
                                self.num_area.transform([selected_area])[0],
                                self.num_item.transform([selected_item])[0],
                                selected_year,
                                area_harvested,
                                yield_value
                            ]], columns=self.encoded_names)
                        
# 2E1 Subsequent times Model training o clicking predct button to avoid processing time on each and every input
                            # Make prediction (will be log-transformed)
                            prediction_log = self.model.predict(input_data)
                            prediction = np.expm1(prediction_log)[0]
                            html_code = """
                            <div style="text-align: center; font-size: 24px;">
                             Predicted Production
                            </div>
                            """
                            st.markdown(html_code, unsafe_allow_html=True)
                            st.metric("Production", f"{prediction:.2f} tonnes")
# 2E2 Feature importance
                            if hasattr(self.model['regressor'], 'feature_importances_'):
                                html_code = """
                                <div style="text-align: center; font-size: 24px;">
                                 Feature Importance
                                </div>
                                """
                                st.markdown(html_code, unsafe_allow_html=True)
                                importance_df = pd.DataFrame({
                                    'Feature': ['Area', 'Crop Type', 'Year', 'Area Harvested', 'Yield'],
                                    'Importance': self.model['regressor'].feature_importances_
                                })
                                fig = px.bar(importance_df, x='Feature', y='Importance',
                                           title='Feature Importance Analysis')
                                st.plotly_chart(fig)
# 2Ec Future predictions
                            if future_predict:
                                if selected_year <= max_yr:
                                    st.info('Year not selected and assumed 2030')
                                    selected_year = future_yr
                                future_years = range(max_yr+1, selected_year+1)
                                future_predictions = []
                                
                                for year in future_years:
                                    future_input = input_data.copy()
                                    future_input['Year'] = year
                                    future_pred_log = self.model.predict(future_input)
                                    future_pred = np.expm1(future_pred_log)[0]
                                    future_predictions.append(future_pred)
                                
                                future_df = pd.DataFrame({
                                    'Year': future_years,
                                    'Predicted Production': future_predictions
                                })
                                fig = px.line(future_df, x='Year', y='Predicted Production',
                                            title=f'Future Production Predictions for {selected_item} in {selected_area}')
                                st.plotly_chart(fig)

                        if st.sidebar.button('Exit'):
                            st.markdown ("### Thank You for using the App")
                            time.sleep(3)
                            # Close streamlit browser tab
                            keyboard.press_and_release('ctrl+w')
                            # Terminate streamlit python process
                            pid = os.getpid()
                            p = psutil.Process(pid)
                            p.terminate()

# 2Ed Data distribution plots
                        # 2Edα Heading Display based on first time or subsequent
                        if st.session_state.first_time == 0:
                            st.session_state.first_time = 1
                            area_df = df
                            item_df = df
                            bar_title = 'All Crops'
                            html_code = """
                            <div style="text-align: center; font-size: 24px;">
                             Data Analysis for all Area and Crops in Dataset
                            </div>
                            """
                            st.markdown(html_code, unsafe_allow_html=True)
                        else:
                            area_df = df[df['Area'] == selected_area]
                            item_df = df[df['Item'] == selected_item]
                            bar_title = selected_item
                            html_code = """
                            <div style="text-align: center; font-size: 24px;">
                             Data Analysis for Area / Item Selected
                            </div>
                            """
                        # 2Edβ Top 15 Area production for the selected crop
                        df1 = item_df.groupby('Area', as_index=False)['Production'].sum()
                        top_15 = df1.sort_values(by='Production', ascending=False).head(15)
                        html_code = """
                        <div style="text-align: center; font-size: 24px;">
                         Top 15 Areas of Production
                        </div>
                        """
                        st.markdown(html_code, unsafe_allow_html=True)
                        fig = px.bar(top_15, x='Area', y='Production',
                                   title=bar_title)
                        st.plotly_chart(fig)

                        # 2Edγ Production distribution by Crop for selected Area
                        st.markdown(html_code, unsafe_allow_html=True)
                        if bar_title != 'All Crops':
                            st.write(f'for Area {selected_area}')
                            st.write('')
                        fig = px.box(area_df, x='Item', y='Production',
                                   title='Production Distribution by Crop')
                        st.plotly_chart(fig)
                        
                        # 2Edδ Area Harvested vs Production
                        fig = px.scatter(area_df, x='Area_harvested', y='Production',
                                       color='Item', title='Area vs Production')
                        st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in Streamlit app: {str(e)}")
            print(f"Error in Streamlit app:  {str(e)}")
## --------------------------------------------------------------------------------------------------------------------------------------- ##

def main(csv_file):
# Invoking the execution
    Task_run = CropProductionClass()
    Task_run.main_ui(csv_file)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Pass Preprocessed CSV file as parameter to run')
