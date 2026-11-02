
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Starbucks Drink Classifier', layout='wide')

st.title('☕ Starbucks Beverage Calorie Classifier')
st.markdown('This application uses a trained Decision Tree Classifier to predict whether a Starbucks drink is **High Calorie** (>150 Calories) or **Low Calorie** (<=150 Calories).')

# 1. Load the data using relative paths
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_starbucks_menu.csv')
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('starbucks_drinkMenu_expanded.csv')
            df.columns = df.columns.str.strip()
            df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')
            df = df.dropna(subset=['Calories'])
            return df
        except FileNotFoundError:
            st.error("Dataset file not found! Please upload 'processed_starbucks_menu.csv' or 'starbucks_drinkMenu_expanded.csv' to your GitHub repository.")
            return None

# 2. Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('decision_tree_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'decision_tree_model.pkl' not found! Please upload it to your GitHub repository.")
        return None

df = load_data()
model = load_model()

if df is not None and model is not None:
    # Sidebar inputs for prediction
    st.sidebar.header('🔍 Predict a Drink')
    
    # Dynamic selection of drink options from the dataset
    beverage_list = sorted(df['Beverage'].unique())
    selected_beverage = st.sidebar.selectbox('Select a Beverage template:', beverage_list)
    
    # Get typical calories for selected beverage to use as default
    default_calories = int(df[df['Beverage'] == selected_beverage]['Calories'].mean())
    
    user_calories = st.sidebar.slider(
        'Adjust Calories:', 
        min_value=0, 
        max_value=int(df['Calories'].max()), 
        value=default_calories
    )
    
    # Make prediction
    prediction = model.predict([[user_calories]])[0]
    prediction_label = '🔥 High Calorie (>150)' if prediction == 1 else '🍃 Low Calorie (<=150)'
    
    # Display Prediction Results
    st.sidebar.subheader('Prediction Result')
    if prediction == 1:
        st.sidebar.error(prediction_label)
    else:
        st.sidebar.success(prediction_label)
        
    # Main panel tabs
    tab1, tab2 = st.tabs(['📊 Exploratory Data Analysis', '📋 Drink Dataset Quick-View'])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Top 5 Drinks with Highest Calories')
            top_calories = df.nlargest(5, 'Calories')
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(top_calories['Calories'], labels=top_calories['Beverage'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(top_calories)))
            st.pyplot(fig)
            
        with col2:
            st.subheader('Calorie Distribution by Beverage Category')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x='Calories', y='Beverage_category', ax=ax, palette='muted')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    with tab2:
        st.subheader('Starbucks Beverage Dataset')
        st.dataframe(df[['Beverage_category', 'Beverage', 'Beverage_prep', 'Calories']])
