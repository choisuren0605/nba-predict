import streamlit as st
import pandas as pd
import joblib
import pickle
import base64
from sklearn.metrics import accuracy_score
# Load the trained model
clf = joblib.load('model.joblib')

# Load the dictionary for categorical encoding
file = open("dict_all.obj", 'rb')
dict_all = pickle.load(file)

# Load the features
with open('features_list.pkl', 'rb') as file:
    features = pickle.load(file)


def date_features(df):   
    df['date'] = pd.to_datetime(df['GAME DATE'])
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['dayofweek']=df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['season'] = df['month'] % 12 // 3 + 1
    df['weekend'] = df['dayofweek'].map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
    return df

def lag_features(df, n_period,cols): 
    columns_to_convert = ['FG%', '3P%', 'FT%']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df[columns_to_convert]=df[columns_to_convert].fillna(0)
    for lag in range(1, n_period + 1):
        for col in cols:      
           
            df[f'match_lag_{col}_{lag}'] = df.groupby(['PLAYER','TEAM'])[col].shift(lag)
            
            groupby_col =['PLAYER','TEAM','MATCH UP']
            df[f'lag_{col}_{lag}'] = df.groupby(groupby_col)[col].shift(lag)
    return df

def rolling_mean_features(df,cols,window_size):
    columns_to_convert = ['FG%', '3P%', 'FT%']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df[columns_to_convert]=df[columns_to_convert].fillna(0)
    for col in cols:
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['PLAYER','TEAM'])[col].rolling(window=window_size, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['PLAYER','TEAM'])[f'match_rolling_mean_{col}_{window_size}'].shift(fill_value=None)
    return df

# Function to preprocess data for prediction
def preprocess_data(df, dict_all):
    cat_cols = ['PLAYER', 'TEAM', 'MATCH UP']
    df[cat_cols] = df[cat_cols].astype(object)
    for col in df[cat_cols]:
        df[col].replace(dict_all[col], inplace=True)
    return df

# Function to make predictions
def make_predictions(model, df, test):

    data=test.copy()
    test['is_test'] = 'yes'
    df = pd.concat([df, test])
    df=df.reset_index(drop=True)
    df['GAME DATE'] = pd.to_datetime(df['GAME DATE'])

    feat_cols=['FGM', 'FGA', '3PM','FG%', '3P%', 'FT%','MIN','3PA', 'FTM', 'FTA','OREB',
               'DREB','REB','AST','STL','BLK','TOV','+/-','PF','FP']

    df = preprocess_data(df, dict_all)
    df=df.sort_values(by=['GAME DATE'])
    df=lag_features(df,10,cols=feat_cols)
    df = rolling_mean_features(df, cols=feat_cols, window_size=2)
    df = rolling_mean_features(df, cols=feat_cols, window_size=3)
    df = rolling_mean_features(df, cols=feat_cols, window_size=4)
    df = rolling_mean_features(df, cols=feat_cols, window_size=5)
    df=date_features(df)
    
    test=df.loc[df['is_test']=='yes']
    test = test[features]
    pred = model.predict_proba(test)
    pred_df = pd.DataFrame(pred, columns=['0-5', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100', '6-10'])
    pred_df=pred_df[['0-5', '6-10','11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']]

    pred_df = pred_df.applymap(lambda x: f'{x:.1%}')
    # pred_df['Max_Label'] = pred_df.idxmax(axis=1)  
    data_columns=['0-5', '6-10','11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']
    pred_df[data_columns] = pred_df[data_columns].apply(lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce') / 100)
    # Find the column with the maximum value for each row
    pred_df['predict'] = pred_df[data_columns].idxmax(axis=1)

    pred_df.insert(loc=0, column='PLAYER', value=data['PLAYER'])
    pred_df.insert(loc=1, column='TEAM', value=data['TEAM'])
    pred_df.insert(loc=2, column='MATCH UP', value=data['MATCH UP'])
    pred_df.insert(loc=3, column='GAME DATE', value=data['GAME DATE'])
    pred_df = pred_df.applymap(lambda x: f'{float(x):.1%}' if pd.to_numeric(x, errors='coerce') == x else x)

    if 'PTS' in data.columns:
        pred_df.insert(loc=3, column='PTS', value=data['PTS'])
        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
        bin_labels = ['0-5', '6-10','11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']
        pred_df['PTS_bin'] = pd.cut(pred_df['PTS'], bins=bins, labels=bin_labels, right=False)
 
    return pred_df

# Streamlit App
def main():
    st.set_page_config(layout="wide")

    def sidebar_bg(side_bg):

        side_bg_ext = 'png'

        st.markdown(
         f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
        }}
        </style>
      """,
      unsafe_allow_html=True,
      )
    side_bg = '11.jpg'
    # sidebar_bg(side_bg)
    st.sidebar.title('NBA Predictions App')
    

    # Upload Main Data
    uploaded_file_df = st.sidebar.file_uploader("1.Main датаг оруулна уу (CSV file)", type=["csv"])

    # Upload Test Data
    uploaded_file_test = st.sidebar.file_uploader("2.Test датаг оруулна уу (CSV file)", type=["csv"])

    if uploaded_file_df is not None and uploaded_file_test is not None:
        df = pd.read_csv(uploaded_file_df)
        test = pd.read_csv(uploaded_file_test)
        
        st.warning("Test Data:")
        st.write(test.head(2))

        # Button to make predictions
        if st.sidebar.button("Predict"):
            predictions = make_predictions(clf, df, test)
            
            st.success("Result:")
            st.markdown(
                f'<style>.css-1jw8d53{{max-width: none !important; width: 100vw;}}</style>',
                unsafe_allow_html=True
            )
            st.data_editor(predictions, num_rows="dynamic") 

            csv_data = predictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Татах",
                data=csv_data,
                file_name="predictions.csv",
                key="download_predictions",
            )

        # Button to make predictions
        if st.sidebar.button("Result"):
            
            predictions = make_predictions(clf, df, test)
            st.success("Result:")
            result=predictions[['PLAYER','PTS_bin','predict']]
            st.data_editor(result, num_rows="dynamic")
            csv_data_res = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Үр дүнг татах",
                data=csv_data_res,
                file_name="result.csv",
                key="download_result",
            )

            accuracy = accuracy_score(predictions['PTS_bin'],predictions['predict'])
            st.text(f"Accuracy: {accuracy:.2f}")
           
if __name__ == '__main__':
    main()

