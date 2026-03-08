import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI 주가 예측", layout="wide")
st.title("📈 AI 국내 주식 분석 터미널")

with st.sidebar:
    st.header("설정")
    target = st.text_input("종목 코드 (6자리)", value="005930")
    days = st.slider("예측 기간", 5, 30, 14)
    run = st.button("분석 실행")

if run:
    try:
        df = fdr.DataReader(target, datetime.now() - timedelta(days=365), datetime.now())
        if not df.empty:
            model = ExponentialSmoothing(df['Close'], trend='add').fit()
            forecast = model.forecast(days)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), name='과거 주가'))
            f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
            fig.add_trace(go.Scatter(x=f_dates, y=forecast, name='AI 예측', line=dict(dash='dash', color='red')))
            
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"예측 완료: {days}일 후 예상가 {forecast.iloc[-1]:,.0f}원")
    except Exception as e:
        st.error(f"오류 발생: {e}")
