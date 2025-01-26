<h1 align="center">
Predicting Crop Production Based on Agricultural Data
</h1>
[ Python Project - Data Cleaning and Preprocessing, Exploratory Data Analysis (EDA), Data Visualization ,SQL, Streamlit, Machine Learning(Regression) Domain: Agriculture ]

### Project Objective
This project aims to develop a regression model that forecasts crop production (in tons) based on agricultural factors such as area harvested (in hectares), yield (in kg/ha), and the year, for various crops grown in a specific region using a FAOSTAT Dataset

### Approach
  1.	Cleansing  and pre-processing the dataset received by .
      • suitably handling the null, missing values, etc and standardizing column metrics.
    	• Filter relevant columns for analysis data.
    	• Consolidating rows by merging and creating values as columns based on column element (Area harvested, Production & Yield)
    	• Understaing outliers and suitably act upon them. 
    	
  3. Perform Exploratory Data Analysis (EDA) to uncover trends, crop distribution and insights and represent them visually using graphs and plots
     
  4. Model Performance - Metrics involving R2, MSE, MAE,etc. comparing different models.

  5.	Predicting Production measured in tons through streamlit application and showcase the predicted value based on selected features.


### Technologies used and Pre-requisites

All details and installation information are provided in the Prerquisite.PDF file.

1.	**Python:** The programming language used for building the application and scripting tasks.

    Visit the URL  [Download Python | Python.org](https://www.python.org/downloads/)    to download and install python in the PC

    **Install necessary Python packages / Libraries:** Install the necessary packages / libraries using pip or conda package manager. Required libraries include DateTime, Regex, Streamlit, Pandas, mysql-connector-python and google-api-python-client

2.	**Streamlit:** Streamlit is an open-source Python framework designed to create and share web applications. It is specifically built for Python, making it easy for data scientists and machine learning engineers to deploy their models and visualize data without needing extensive knowledge of web development.
	
        A web-based user interface is developed using Streamlit for Sql queries & result display.

### Executing the Application 

go through and to run the project, follow these steps:

1.	Install Python and required packages / libraries as given above
2.	Download the Python Applications from my this Git hub repositary.
5.	Launch the Application through Jupyter lab's Terminal (Windows Powershell) typing.  
     **Python Jag_cropPredict_proj**

### Features

The application offers the following features:
+	Cleansing & storing (in MySql) the given dataset/s.
+	Merging Datasets as FinalMerged and storing it in MySql for Power BI visualization and query analysis.
+	Data analysis and visualization through charts and graphs using Power BI.
+	Analysis of details from SQL database using SQL queries for a ten different pre-defined search options.

### References

> Python Documentation: [https://docs.python.org/](https://docs.python.org/)
> Streamlit Documentation: https://docs.streamlit.io/

#### My Contact
📧 Email: jagdeeshv@hotmail.com  
🌐 LinkedIn: https://www.linkedin.com/in/jagadeesh-v-a09a681b7 (Demo video will be uploaded shortly)

For any further questions or inquiries, feel free to reach out. I am happy to assist you with any queries.

