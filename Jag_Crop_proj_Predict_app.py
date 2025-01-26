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

class CropProductionPredictor:
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

    def Merge_Element_data(self, csv_file):
        try:
            print ('In Merge Data')
            df = pd.read_csv(csv_file)
            # cols are 'Area', 'Item', 'Element', 'Year', 'Value'
            # Make values of Areea_harvested, Yield & Production in single rows for easy and better analysis & prediction 
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
            
            # Removing outliers using IQR method
            def remove_outliers(df, column):
                print ('In Outliers')
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
            print(f"Error details: {str(e)}")
            return None
    
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
            print(f"Error details: {str(e)}")
            return None, None
    
    def train_model(self, x, y):
        print ('In Train model')
        try:
            if x is None or y is None:
                raise ValueError("Features or target is None")
            
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(x_train, y_train)
            
            # Make predictions
            y_pred_log = self.model.predict(x_test)
            
            # Transform predictions back to original scale
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
            print(f"Error details: {str(e)}")
            return None
    
    def main_ui(self):
        try:
            st.title('Crop Production - Prediction')
            print('--------------------------------------------------------------------------------------\n')
            if 'cnt' not in st.session_state:
                st.session_state.cnt = 0

            csv_file = st.file_uploader("Upload FAOSTAT Preprocessed CSV file", type=['csv'])
            
            if csv_file is not None:
                df = self.Merge_Element_data(csv_file)

                if df is not None and not df.empty:
                    if st.session_state.cnt == 0:
                        st.session_state.cnt = 1
                        x, y = self.prepare_features(df)
                        if x is not None and y is not None:
                            metrics = self.train_model(x, y)
                            
                            if metrics is not None:
                                html_code = """
                                <div style="text-align: center; font-size: 24px;">
                                 Model Performance of entire dataset
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

                    # Input parameters for prediction
                    st.sidebar.header('Input Parameters')

                    selected_area = st.sidebar.selectbox('Select Area', df['Area'].unique())
                    filtered_df = df[df['Area'] == selected_area]
                    selected_item = st.sidebar.selectbox('Select Crop', filtered_df['Item'].unique())

                    # Getting median values from similar crops/areas for Area Harvested & Yield as default
                    default_area = df[(df['Item'] == selected_item) & (df['Area'] == selected_area)]['Area_harvested'].median()
                    default_yield = df[(df['Item'] == selected_item) & (df['Area'] == selected_area)]['Yield'].median()
                    area_harvested = st.sidebar.number_input('Area_harvested (ha)', min_value=0.0, value=float(default_area))
                    yield_value = st.sidebar.number_input('Yield (kg/ha)', min_value=0.0, value=float(default_yield))

                    future_predict = st.sidebar.checkbox('Future Predictions (2024-2030)')
                    min_yr = df['Year'].min()
                    max_yr = df['Year'].max()
                    future_yr = round(max_yr, -1)
                    if future_predict:
                        if future_yr <= max_yr:
                            future_yr = future_yr+ 10
                    selected_year = st.sidebar.slider('Select Year', min_yr, future_yr, max_yr)

                    print(f"1. Area {selected_area}")
                    print(f"1. Item {selected_item}")
                    print(f"1. Year {selected_year}")
                    print(f"1. Harvested {area_harvested}")
                    print(f"1. Yield {yield_value}")
                    print(f"1. Future predict {future_predict}\n")

                    df_area_item = df[(df['Item'] == selected_item) & (df['Area'] == selected_area)]
                    df_area_item.to_csv('D:\Guvi\Crop Proj\FAOSTAT_data\check.csv', index=False)
                    df_area = df[df['Area'] == selected_area]

                    x, y = self.prepare_features(df_area_item)
                    print(x)
                    if x is not None and y is not None:
                        metrics = self.train_model(x, y)
                        
                        if metrics is not None:
                            if st.sidebar.button('Predict'):
                                print('\nPredict Pressed')
                                input_data = pd.DataFrame([[
                                    self.num_area.transform([selected_area])[0],
                                    self.num_item.transform([selected_item])[0],
                                    selected_year,
                                    area_harvested,
                                    yield_value
                                ]], columns=self.encoded_names)
                                
                                print(f"2. Area {selected_area}")
                                print(f"2. Item {selected_item}")
                                print(f"2. Year {selected_year}")
                                print(f"2. Harvested {area_harvested}")
                                print(f"2. Yield {yield_value}")
                                print(f"2. Future predict {future_predict}\n")
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
############                                
                                # Feature importance
                                if hasattr(self.model['regressor'], 'feature_importances_'):
                                    print('\nhasattr in')
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

                                # Show future predictions
                                if future_predict:
                                    if selected_year <= max_yr:
                                        st.info('Year not selected and assumed 2030')
                                        selected_year = future_yr
                                    print(f"3. Year {selected_year}")
                                    future_years = range(max_yr+1, selected_year+1)
                                    print(f"Future yr {future_years}")
                                    future_predictions = []
                                    
                                    for year in future_years:
                                        print(f"Prcs yr {year}")
                                        future_input = input_data.copy()
                                        future_input['Year'] = year
                                        future_pred_log = self.model.predict(future_input)
                                        future_pred = np.expm1(future_pred_log)[0]
                                        future_predictions.append(future_pred)
                                    
                                    future_df = pd.DataFrame({
                                        'Year': future_years,
                                        'Predicted Production': future_predictions
                                    })
                                    print(f"Future df {future_df}")
                                    fig = px.line(future_df, x='Year', y='Predicted Production',
                                                title=f'Future Production Predictions for {selected_item} in {selected_area}')
                                    st.plotly_chart(fig)

                            # Data distribution plots
                            html_code = """
                            <div style="text-align: center; font-size: 24px;">
                             Data Analysis
                            </div>
                            """
                            st.markdown(html_code, unsafe_allow_html=True)
                            fig = px.box(df_area, x='Item', y='Production',
                                       title='Production Distribution by Crop')
                            st.plotly_chart(fig)
                            
                            fig = px.scatter(df_area, x='Area_harvested', y='Production',
                                           color='Item', title='Area vs Production')
                            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in Streamlit app: {str(e)}")
            print(f"Error details: {str(e)}")

def main():
    predictor = CropProductionPredictor()
    predictor.main_ui()

if __name__ == "__main__":
    main()
