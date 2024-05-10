from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pandas_datareader as data
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime
from keras.models import load_model
import streamlit as st

# start='2010-01-01'
# end='2021-12-31'
startdate=datetime(2010,1,1)
enddate=datetime(2023,1,1)

with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Home","Trend Analysis","About","Help"]
    )



if selected=="Home":

    st.title('Price Prediction')
    user_input=st.text_input('Enter Stock Ticker','AAPL')
    # df=data.DataReader(user_input,'yahoo',start,end)
    df=pdr.get_data_yahoo(user_input,start=startdate,end=enddate)

    #describing data to user
    st.subheader('Past 10years data')
    st.write(df.describe()) 

    #visuaization
    st.subheader(f'Time Chart of {user_input}')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # st.subheader('Closing Price vs Time Chart with 100MA Days (Moving Average) ')
    ma100=df.Close.rolling(100).mean()
    # fig=plt.figure(figsize=(12,6))
    # plt.plot(ma100)
    # plt.plot(df.Close)
    # st.pyplot(fig)

    st.subheader('Closing price with moving avg ')
    ma200=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig) 

    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    data_training_array=scaler.fit_transform(data_training)
    # print(data_training_array)
    # print(data_training_array.shape)


    # loading my modelpy
    model =load_model('keras_model.h5')

    #testing 
    past_100_days=data_training.tail(100)
    #
    # final_df=past_100_days.append(data_testing,ignore_index=True)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)


    input_data=scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]
    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100 :i])
        y_test.append(input_data[i,0])

    x_test,y_test=np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)
    scaler=scaler.scale_
    scale_factor=1/scaler[0]
    y_predicted=y_predicted*scale_factor
    y_test=y_test*scale_factor

    #final graph 
    st.subheader('Predictions vs Original')
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price') 
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

if selected=="Trend Analysis":
    from datetime import date
    from prophet import Prophet
    from prophet.plot import plot_plotly
    from plotly import graph_objs as go
    start="2010-01-01"
    today=date.today().strftime("%Y-%m-%d")

    st.title("Stock Trend Analysis")

    stock={"AAPL","GOOG","MSFT"}
    # selected_stock=st.selectbox("Choose Stock",stock)
    selected_stock=st.text_input("Enter stock","AAPL")
    n_years=st.slider("Years of prediction",1,4)
    period=n_years*365

    @st.cache
    def load_data(ticker):
        data=yf.download(ticker,start,today)
        data.reset_index(inplace=True)
        return data

    data_load_state=st.text("Load data...")
    data=load_data(selected_stock)
    data_load_state.text("Loading data ..done!!")

    st.subheader(f'Raw Data of {selected_stock}')
    st.write(data.tail())

    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="stock_close"))
        fig.layout.update(title_text='Time serise data',xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df_train=data[['Date','Close']]
    df_train=df_train.rename(columns={'Date':"ds","Close":"y"})

    m=Prophet()
    m.fit(df_train)
    future=m.make_future_dataframe(period)
    forecast=m.predict(future)

    st.subheader(f'Forecast data {selected_stock}')
    st.write(forecast.tail())

    st.write("Forecast data")
    fig1=plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.write('Forecast components')
    fig2=m.plot_components(forecast)
    st.write(fig2)

    
if selected=="About":
    st.title("About")
    st.write("Welcome to our Stock Price Prediction and Trend Analysis project! We leverage cutting-edge technology and advanced algorithms to provide accurate predictions and insightful analysis for stock market enthusiasts and investors.")
    st.subheader("How It Works?")
    st.write("Our stock price prediction and trend analysis system employ a combination of machine learning, statistical models, and historical data to generate forecasts and identify meaningful patterns. Here's an overview of our process:")
    st.write('''
        1. **Data Collection**: We gather extensive historical stock market data, including prices, trading volumes, and other relevant metrics, from reliable sources.
        2. **Data Preprocessing**: Our algorithms process and clean the data, ensuring accuracy and removing any inconsistencies or outliers.
        3. **Feature Engineering**: We extract meaningful features from the data, which are crucial for training our prediction models and identifying trends.
        4. **Model Training**: We utilize state-of-the-art machine learning techniques to train predictive models on the processed data. These models learn from historical patterns and correlations to predict future stock prices.
        5. **Prediction and Analysis**: Once the models are trained, we apply them to new data to generate predictions and perform trend analysis. We provide users with easy-to-understand visualizations and reports that highlight important insights and trends.
''')
    st.subheader("Key Features")
    st.write('''
    - **Accurate Predictions**: Our models have been rigorously trained and tested to ensure accurate predictions for a wide range of stocks.
    - **Trend Analysis**: We analyze historical data and identify trends that can help users understand the market dynamics and make informed decisions.
    - **Visualization**: We provide intuitive charts, graphs, and visualizations that make it easy to interpret the predicted stock prices and trends.
    - **Customization**: Users can customize their analysis by selecting specific stocks, time periods, and other relevant parameters.
    - **Real-time Updates**: We strive to provide timely updates and incorporate the latest market data to enhance the accuracy of our predictions.

    ''')


    st.subheader("Contact Us")
    st.write("If you have any questions, feedback please don't hesitate to contact us. Our team is here to assist you and ensure you have a seamless experience using our platform.")


if selected=="Help":
    st.title("Help!")
    st.write("If you dont know ticker of the company you looking for click on the link (https://finance.yahoo.com/)")
    st.write("Deep analysis of stock click the link (https://tradingview.com/)")
    st.write("Basic Stock market course by zerodha click on the link (https://zerodha.com/varsity/)")
