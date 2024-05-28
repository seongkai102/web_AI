import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import requests
import io

# github 파일 다운로드 링크
github_raw_link = 'https://raw.githubusercontent.com/seongkai102/web_AI/main/data.csv'

# 파일 다운로드 함수
def download_file(url):
    response = requests.get(url)
    return response.content

# 파일 다운로드
csv_content = download_file(github_raw_link)

# 다운로드한 파일 불러오기
df = pd.read_csv(io.BytesIO(csv_content), encoding='cp949')

df['일자'] = pd.to_datetime(df['일자']).astype(np.int64) // 10**9 

df_shifted = df.shift(fill_value=0)

X = df_shifted[['일자', '평균가격', '수입량','평년반입량(KG)','평년반입량증감률(%)']]
y = df_shifted['평균가격']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

def predict_price(model, date_str):
    date = pd.to_datetime(date_str)
    date_int = date.timestamp() * 1000
    features = [[date_int, 0, 0, 0, 0]]
    predicted_price = model.predict(features)
    return predicted_price[0]

input_date = '2020-02-01'
predicted_price = predict_price(model, input_date)

###################################################

with st.sidebar: #사이드 바
    choose = option_menu("Team 새싹's", ["인공지능 예측", "평균가격 그래프"],
        menu_icon="bi bi-stars", default_index=0,
        styles={
        "container": {"padding": "5!important", "background-color": "#ebc659"},
        "icon": {"color": "black", "font-size": "25px"},        
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee",
                    "color": "black"},  # 텍스트 색상을 여기서 변경
        "nav-link-selected": {"background-color": "#3ae0ca"},
        }
        )


now= datetime.now()

if now.year <= 2024:
    td = int(f'{now.year-1}{now.month:02}{now.day:02}') #:02 칸 2개 생성
else:
    st.title('추후에 업데이트 예정입니다')

#파일 다시 불러오기   
df = pd.read_csv(io.BytesIO(csv_content), encoding='cp949')
#열삭제 
df = df.drop(columns=["품목명", "수입량", "평년반입량(KG)", "평년반입량증감률(%)"])

#날짜 옆 값 추출
found = False
result = "N/A"

for index, row in df.iterrows():
    if row['일자'] == td:
        result = row['평균가격']  
        # print(f"일자가 '{td}'인 행의 인덱스: {index}, 옆 열 값: {result}")
        found = True
        break
if not found:
    print(f"일자가 '{td}'인 행을 찾을 수 없습니다.")
    
result = np.asarray(result, dtype=float)
#출력 부분
if choose == "인공지능 예측":
    st.session_state.page = "home" 
     
    st.title("인공지능 가격예측")
    st.subheader("오늘의 감자 예측가격")
    st.subheader(f"{now.year}년 {now.month}월 {now.day}일 :green[{round(predicted_price)}]원")
    st.text("(감자 KG당 가격)")
    st.title("")
    st.metric(label="작년과 오늘의 가격차이", value=f"예측가격 {round(predicted_price)}원", 
              delta=f"{round(predicted_price - result)} ₩") 
    st.write(':red[빨간], :green[초록] 글씨가 가격차이입니다.')
    
elif choose == "평균가격 그래프":
    st.session_state.page = "page1"

    st.title("평균가격 그래프")
    
    st.text("그래프 출력시 시간이 소요됩니다.")
    st.text("조금만 기다려 주세요.")

    df['일자'] = pd.to_datetime(df['일자'], format='%Y%m%d')

    # 그래프 그리기
    fig, ax = plt.subplots()
    ax.plot(df['일자'], df['평균가격'], color='blue')  # 마커 제거
    ax.set_xlim(pd.Timestamp('2019-01-03'), pd.Timestamp('2023-12-30'))  # x축 범위
    ax.set_ylim(0, 100000)  # y축 범위
    ax.set_xlabel('date')  # x 라벨
    ax.set_ylabel('price(KRW, ₩)')  # y 라벨
    ax.set_title("Average Price")  # 그래프 이름
    plt.xticks(rotation=45)  # x축 레이블 회전
    # 그래프 표시
    plt.tight_layout()
    st.pyplot(fig)
