import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy as np, pandas as pd, matplotlib.pyplot as plt, keras, itertools
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import plotly.graph_objs as go


app = dash.Dash()
server = app.server

Dataset = pd.read_csv("Test_Weekly_Activity_Score_R_local.csv")

df=Dataset[["repoID", "Normalised_Activity_Score"]]

model=load_model("Spring_2021_Model3.h5")

rows = [108110, 17165658, 2446718, 99919302, 206417, 45721011, 31006158, 39464018, 462292, 118030974, 21193524, 164026325, 151294653, 102083576, 108125, 832680, 81355383, 132600378, 206412, 32199982, 32848140, 19244865, 70746484, 93486496, 108050, 143965255, 4719160, 9735077, 37115105, 108051, 699689, 7437073, 206422, 309009, 23876346, 206402, 149178674, 143887688, 10223615, 71975377, 11151771, 153295119, 52028643, 19514152, 87100707, 41952293, 81920458, 33294317, 27557391, 45165994, 58072252, 58073483, 87100729, 206370, 143950, 63044524, 113041753, 35964690, 158448038, 16021499, 93444591, 131209056, 45896813, 9390430, 52027975, 131394823, 137439053, 116109319, 15741777, 146502453, 135523369, 83877385, 7567432, 3658431, 17443202, 136240779, 107435578, 100294206, 10860602, 75708591, 20675635, 15881868, 33679984, 93211371, 154606109, 131209340, 82691991, 74950506, 183648903, 32376872, 206478, 105323132, 61863697, 126176628, 38525117, 82371321, 149782046, 133959957, 64581179, 46900076, 33397227, 52050228, 37991233, 107435963, 7770260, 92971378, 96584339, 31976270, 71356594, 68686441, 404397, 108511089, 70037034, 98013452, 60309847, 138208400, 10860597, 53160128, 106789422, 10860607, 113678287, 161004, 107435933, 102083574, 107435477, 61801701, 113564038, 63669374, 95679812, 18627124, 13853693, 6898377, 107435631, 113564089, 60121077, 63879166, 122675902, 30724109, 206357, 107435691, 88242107, 107435948, 125097454, 55078765, 55437297, 26353925, 9759451, 113564280, 107435906, 113571428, 73428254, 113678093, 63399426, 20028272, 123161718, 113488245, 107433731, 65358491, 107433033, 132148763, 29979177, 31933589, 206635, 107433355, 107433968, 107432816, 92187408, 71358594, 113564765, 109810936, 11362691, 46776191, 63879003, 1514767, 60209452, 29979115, 87100755, 112034937, 47363193, 41820544, 107433375, 92714730, 107434610, 113677914, 107434396, 113678030, 154605982, 102083575, 107436041, 125097486, 24584820, 14149824, 91362946, 34020308, 111739948, 105204510, 113484173, 31976263, 11304845, 116926787, 113484230, 20248083, 22498765, 83632281, 14740847, 6898380, 16795933, 107436427, 6898383, 33884891, 50904245, 2211243, 138754790, 86057409, 41348335, 117599238, 109678056, 147531218, 144622232, 52588453, 99279793]


#app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(style={'backgroundColor': 'white'},children=[
html.P(html.H1(children='GitHub Repository Health Forecast',style={
            'textAlign': 'center',
            'height': '60px',
    'line-height': '60px',
    'border-bottom': 'thin lightgrey solid',
            'color': 'black'
        })),
html.Label('Repository ID',style={
            'color': 'black',
            'font-size':25
        }),
dcc.Dropdown(id = 'g3',options=[{
                         'label' : i, 
                         'value' : i
                 } for i in rows],style = dict(
                            width = '30%',
                            display = 'inline-block',
                            verticalAlign = "middle"
                            )),
dcc.Graph(id='example')
])
 

'''
@app.callback(Output(component_id='submit-val', component_property='fig'), 
              [Input(component_id='g3', component_property='value')])'''

@app.callback(
    dash.dependencies.Output('example', 'figure'),
    [dash.dependencies.Input('g3', 'value')]
)

def update_graph(value):
    #Dataset = pd.read_csv("Test_Weekly_Activity_Score_R_local.csv")
    #model=load_model("Spring_2021_Model3.h5")
    #print(value)
    value = int(value)
    repo_data=df[df['repoID']==value]
    single_repo_series=repo_data.Normalised_Activity_Score
    Risk_Score=single_repo_series[-20:].values
    series = np.array(Risk_Score)
    series=np.reshape(series,(1,series.shape[0],1))

    predictions = np.zeros(12)
    predictions[0] = model.predict(series, batch_size = 1)
    n_ahead = 12
    if n_ahead > 1:
        for i in range(1,n_ahead):
            x_new = np.append(series[0][1:],predictions[i-1])
            series = x_new.reshape(1,x_new.shape[0],1)
            predictions[i] = model.predict(series,batch_size = 1)
            
    Final_Score=predictions.reshape(12)
    Final_Score[Final_Score<0] = 0

        # data to be plotted 
    x = np.arange(start=1, stop=13, step=1)
    first = go.Scatter(x=x, y=Final_Score)
    data = [first]
        #print(i)
        #print(rows[i])
    #fig = plt.figure()
    #fig.patch.set_facecolor('black') 
    #plt.title("Activity Score for the next 12 weeks")  
    #plt.xlabel("Date")  
    #plt.ylabel("Activity-Score")
    #plt.plot(x, Final_Score, color ="green") 
    fig = go.Figure(data)
    return fig


if __name__=='__main__':
    model=load_model("Spring_2021_Model3.h5")
    Dataset = pd.read_csv("Test_Weekly_Activity_Score_R_local.csv")
    app.run_server(host='0.0.0.0', port=80)