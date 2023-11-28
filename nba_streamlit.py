import streamlit as st
import pandas as pd
import joblib
import pickle
import base64
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
    pred_df = pd.DataFrame(pred, columns=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-100'])
    pred_df = pred_df.applymap(lambda x: f'{x:.1%}')
    pred_df.insert(loc=0, column='PLAYER', value=data['PLAYER'])
    pred_df.insert(loc=1, column='TEAM', value=data['TEAM'])
    pred_df.insert(loc=2, column='MATCH UP', value=data['MATCH UP'])
    pred_df.insert(loc=3, column='GAME DATE', value=data['GAME DATE'])
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
    uploaded_file_test = st.sidebar.file_uploader("2.Test датаг оруулна уу (Excel file)", type=["xlsx"])

    if uploaded_file_df is not None and uploaded_file_test is not None:
        df = pd.read_csv(uploaded_file_df)
        test = pd.read_excel(uploaded_file_test)
        
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
                label="Download",
                data=csv_data,
                file_name="predictions.csv",
                key="download_predictions",
            )
if __name__ == '__main__':
    main()
