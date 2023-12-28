#!/usr/bin/env python
# coding: utf-8

# # bees.csv를 dash 시각화
# 
# 

# #### 0. py 파일 실행 결과

# #### 1. 라이브러리 설치 및 불러오기

# In[1]:


import os
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)


# #### 2. 데이터 불러오기 및 전처리

# In[4]:


app = Dash(__name__)

# 데이터 불러오기 및 전처리
df=pd.read_csv('./data/intro_bees.csv')
df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
df.head()


# #### 3. 앱 레이아웃

# In[5]:


# App layout

app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}), #H1 -> 웹페이지의 제목, 페이지 중앙 정렬

    dcc.Dropdown(id="slct_year",
                 options=[ # 실제로 사용자에게 보여지는 부분
                     {"label": "2015", "value": 2015}, #데이터프레임의 YEAR 변수가 정수여서 정수로 넣어줌. VALUE값은 항상 데이터베이스, API, 데이터프레임에서 변수의 데이터 타입과 같아야함.
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]), #해당 children 부분은 없어도 된다.
    html.Br(), # DIV와 그래프 사이의 공백 추가

    dcc.Graph(id='my_bee_map', figure={})

])

#-------------------------------------------------------------------------------------
# 전체 대시보드를 생성하려면 dash 구성 요소를 앱 레이아웃인 내부에 있는 그래프와 연결해야함
# 콜백을 사용하여 이를 연결해야함.


# In[6]:


# Connect the Plotly graphs with Dash Components
# 콜백에는 아웃풋과 인풋이 있음, 해당 Task에선 2개의 출력과 1개의 입력
@app.callback(
    [Output(component_id='output_container', component_property='children'), #콜백은 아이디와 구성요소를
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd): # 콜백의 함수 정의, value 값 한개가 들어감. 값을 여러개 하고 싶다면 레이아웃에서 multi = True로 변경
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd) #컨테이너 내부 출력,

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    dff = dff[dff["Affected by"] == "Varroa_mites"] # 필터링된 데이터 프레임 생성

    # Plotly Express
    fig = px.choropleth( #등치맵
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

    # Plotly Graph Objects (GO)
    # fig = go.Figure(
    #     data=[go.Choropleth(
    #         locationmode='USA-states',
    #         locations=dff['state_code'],
    #         z=dff["Pct of Colonies Impacted"].astype(float),
    #         colorscale='Reds',
    #     )]
    # )
    #
    # fig.update_layout(
    #     title_text="Bees Affected by Mites in the USA",
    #     title_xanchor="center",
    #     title_font=dict(size=24),
    #     title_x=0.5,
    #     geo=dict(scope='usa'),
    # )

    return container, fig # 콜백의 아웃풋이 2개이므로 2개의 인수를 반환해야함.


# #### 4. 대시보드 구현


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




