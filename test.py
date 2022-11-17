#import streamlit for web_app 
from json import load
import streamlit as st
#import date for the time_period
from datetime import date

#import yahoo_finance to fetch data
import yfinance as yf
#import prophet for forecasting 
from prophet import Prophet
from prophet.plot import plot_plotly

#import graph as objects from plotly to visualise
from plotly import graph_objs as go

#for working with csv
import csv

import yahoo_fin.stock_info as si


r=open('stock.csv','r',encoding='utf-8')
reader=csv.reader(r)

people=[]

for row in reader:
    people.append(row)
    
  
stock_list=[]  
for item in people:
    stock_list.append(item[0])
    #stock_list.append(item[1])
    


# print(stock_list)


#put the start date and current date
START = "1990-01-01"
# TODAY = "2016-02-02"

TODAY=date.today().strftime("%Y-%m-%d")

#title of web_app
st.title("Stock Market Prediction Application")

#stock options available to predict
# stock_list = []
# f = open('allStocks.txt', 'r')
# stock_list = f.readlines()
# f.close()

# stocks = ("AAPL", "GOOG", "TWTR", "MSFT", "ORCL", "^IXIC", "SBIN.BO")
#dataset selection
selected_stocks = st.selectbox("Select Dataset for Prediction", stock_list) #selectbox will assign a value to that variable


# stock_data = si.get_quote_table(selected_stocks)
# # st.write(stock_data)
# # st.write(type(stock_data))
# st.subheader("PE Ratio:")
# st.write(stock_data['PE Ratio (TTM)'])

# st.subheader("EPS (TTM)")
# st.write(stock_data['EPS (TTM)'])


#create slider to select predicted year
n_years = st.slider("Days of Prediction...", 1 , 365 )
period = n_years * 1

#load stock data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Load data...")

data = load_data(selected_stocks)
data_load_state.text("Results are...")

#calling raw data
st.subheader('Raw data')
st.write(data)

#plotting raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()
#forcasting using prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast)



#load stock data
@st.cache
def load_data2(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data



data2 = load_data2(selected_stocks)


#calling raw data
st.subheader('test data')
st.write(data2)

st.subheader('Chart based on forecast data')
fig1 = plot_plotly(m, forecast)
st.write(fig1)

st.subheader('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)