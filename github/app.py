import os
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np


from dash.dependencies import Input, Output, State

script_dir = os.path.dirname(__file__)
cwd = os.getcwd()

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = [
    ['olivers', 'navy']
]

app = dash.Dash()
server = app.server
app.config.suppress_callback_exceptions = True

'''
auth = dash_auth.BasicAuth(app
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
'''

colors = {
    'background': '#111111',
    'text': '#0000FF'
}

import pickle

###############################################################################################
####TAB 3 Data#################################################################################
catBoostTable = pd.read_csv('MAE_Trial1/catBoostTableV2.csv')
hyperParameters = pd.read_csv('MAE_Trial1/hyperParameters.csv')
bestCrossValidation = pd.read_csv('MAE_Trial1/bestCV.csv')
with open('MAE_Trial1/featureImportanceV2.pkl', 'rb') as handle:
    featureImportance = pickle.load(handle)
with open('MAE_Trial1/featureInteractionV2.pkl', 'rb') as handle:
    featureInteractions = pickle.load(handle)

#Add a column that contains the text for the hover boxes
catBoostTable['text'] = catBoostTable['Incident Type'] + \
                    '<br>Date & Hour of Incident: ' + catBoostTable['month'] + ' ' + catBoostTable['day'].astype(str) \
                    + ', ' + catBoostTable['hour'].astype(str) + ' oclock' + \
                    '<br> Actual Response Time: ' + catBoostTable['ActualResponseTime'].astype(str) + \
                    '<br> Predicted Response Time: ' + catBoostTable['PredictedResponseTime'].astype(str) + \
                    '<br> Residual: ' + catBoostTable['Residuals'].astype(str)

catBoost_df = catBoostTable.copy().sample(n = 10000, random_state = 42)

#Create the lines for the train and test cross validation
CVx = list(range(1,len(bestCrossValidation['iteration'])+1,1))
#print(CVx)
x_rev = CVx[::-1]

# Line 1
test = bestCrossValidation['test-MAE-mean'].values.tolist()
test_upper = [test[i]+np.multiply(2,bestCrossValidation['test-MAE-std'].values).tolist()[i] for i in range(len(test))]
test_lower = [test[i]+np.multiply(-2,bestCrossValidation['test-MAE-std'].values).tolist()[i] for i in range(len(test))]
test_lower = test_lower[::-1]

# Line 2
train = bestCrossValidation['train-MAE-mean'].values.tolist()
train_upper = [train[i]+np.multiply(2,bestCrossValidation['train-MAE-std'].values).tolist()[i] for i in range(len(train))]
train_lower = [train[i]+np.multiply(-2,bestCrossValidation['train-MAE-std'].values).tolist()[i] for i in range(len(train))]
train_lower = train_lower[::-1]

trace1 = go.Scatter(
    x=CVx+x_rev,
    y=test_upper+test_lower,
    fill= 'tozerox',
    fillcolor='rgba(98,13,14,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    name='Test',
)
trace2 = go.Scatter(
    x=CVx+x_rev,
    y=train_upper+train_lower,
    fill='tozerox',
    fillcolor='rgba(6,10,243,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Train',
    showlegend=False,
)

trace3 = go.Scatter(
    x=CVx,
    y=test,
    line=dict(color='rgb(98,13,14)'),
    mode='lines',
    name='Test',
)
trace4 = go.Scatter(
    x=CVx,
    y=train,
    line=dict(color='rgb(6,10,243)'),
    mode='lines',
    name='Train',
)

catBoostCVData = [trace1, trace2, trace3, trace4]


###############################################################################################
####TAB 2 Data#################################################################################

# Open the features from the model
with open('features.pickle_v14', 'rb') as handle:
    features = pickle.load(handle)
# Delete all features that are not at least 5% as important as the most important feature
features = {key: val for key, val in features.items() if val >= 5}

# Open the Deviance Analysis
devianceAnalysis = pd.read_pickle('testSetDevD.pkl')

# Open the models predictions, which contains the data for the visualizations for the 2nd tab
modelAnalysis = pd.read_pickle('modelAnalysisD.pkl')

mapBoxKey = 'pk.eyJ1IjoibWlrZXF1YW4iLCJhIjoiY2psMzBuNW05MDJ4bDNxcngzdmZqa3R3diJ9.8sYaudSMNUutIvkKwqJS0Q'

###############################################################################################
####TAB 1 Data#################################################################################

#Import the training set, which contains the data for the visualizations in tab 1
#FUTURE WORK:  turn this into df instead of train_set to conserve resources
train_set = pd.read_csv('train_setD.csv')
#drop the rows that do not have a dispatch to arrival time (i.e. not applicable for visualization purposes)
train_set = train_set.dropna(subset=['dispatch_to_arrival_time'])
#Add a column that contains the text for the hover boxes
train_set['text'] = train_set['ADDRESS_X'] + \
                    '<br>Date & Time of Incident: ' + train_set['CREATE_TIME_INCIDENT'].astype(str) + \
                    '<br>Incident: ' + train_set['INCIDENT_TYPE_ID'].astype(str) + ': ' + train_set[
                        'INCIDENT_TYPE_DESC'].astype(str) + \
                    '<br>Dispatch to Arrival Time: ' + train_set['dispatch_to_arrival_time'].astype(str)

#The SDET incident type describes (what I think) is an officer calling in an event that they responded to
#while off duty.  These have extremely high dispatch-to-arrival times and are overall not useful for this model,
#At the same time, transfer the train_set to df (may omit this later to conserve resources)
df = train_set.drop(train_set[train_set.INCIDENT_TYPE_ID == 'SDET'].index)


#There will be a drop down for the users to select what year's data they want to view.  This list provides the input
availableYears = np.sort(df['year'].unique()).tolist()
availableYears.append('All Years')

#There will also be a multiselect for users to see data based on incident types.  This list provides the input
availableIncidents = np.sort(df['INCIDENT_TYPE_ID'].unique()).tolist()

#There will be a box plot/histogram view for dispatches by time of day.  Create a list that facilitates the ordering
timeOfDay = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

#For the Box Plot and Histograms plots, we will use the dictionary below to select the appropriate columns and
# their potential values depending on user inputs.
details =  pd.DataFrame({
    'bTitles':['Box Plots of Dispatch Times by Incident Type',
               'Box Plots of Dispatch Times by Hour Type',
               'Box Plots of Dispatch Times by Month (0 = January, 11 = December)',
               'Box Plots of Dispatch Times by Day of the Week (0 = Monday, 6 = Sunday)',
               'Box Plots of Dispatch Times by Neighborhood'],
    'hTitles':['Distribution by Incident Type',
               'Distribution by Time of Day',
               'Distribution by Month (0 = January, 11 = December)',
               'Distribution by Day of the Week (0 = Monday, 6 = Sunday)',
               'Distribution by Neighborhood'],
    'colNameInDF':['INCIDENT_TYPE_ID','hour','month','dayofweek','NEIGHBORHOOD'],
    'xAxisValues':[
        [],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6],
        []
    ]
    },
    index = ['IT','TOD','M','DOW','N']

)
#print(details)




#print(df.head())



####################################################################################################################
####################################################################################################################
####################################################################################################################

#The base app function will primarily call the two tabs.  If future analytics need to be integrated, this is where
# to create further tabs
app.layout = html.Div([
    html.H1('Cincinnati Police Dispatch Time Analysis',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
            ),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Data Exploration', value='tab-1'),
        #dcc.Tab(label='Model: Response Time', value='tab-2'),
        dcc.Tab(label = 'CatBoost Model: Response Time', value = 'tab-3'),
    ]),
    html.Div(id='content')
])


@app.callback(Output('content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            #style={'backgroundColor': colors['background']},
            children = [
            html.H2(
                'Cincinnati 9-1-1 Calls: Dispatch to Arrival Effectiveness',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),
            html.P(''' Use the visualizations below to analyze the response times for 9-1-1 dispatches.  Response time \
            is measured from the time of reported arrival minus the time of dispatch, and is measured in minutes.  \
            The data depicted on the map below is a random sampling of 10,000 dispatch calls in Cincinnati from \
            October 2014- March 2019.  The box plots and histogram provide further analysis of the \
            data presented on the map.'''),
            html.Div([
                html.P('Filters for Year and Incident Type(s):'),
                dcc.RadioItems(
                    id='yearSelector',
                    options=[{'label': i, 'value': i} for i in availableYears],
                    value='All Years',
                    labelStyle={'display': 'inline-block',
                                # 'font': colors['text']
                                }
                ),
                dcc.Dropdown(
                    id='incident_type',
                    options=[{'label': i, 'value': i} for i in availableIncidents],
                    multi=True,
                    placeholder="Select incident type(s)",
                    value=[]
                )
            ], style={
                'display': 'inline-block',
                'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px 5px'
            }),
            html.Div([
                dcc.Graph(
                    id='Cincinnati 9-1-1 Calls: Dispatch to Arrival Effectiveness',
                    figure={
                        'data': [
                            go.Scattermapbox(
                                lon=df['LONGITUDE_X'],
                                lat=df['LATITUDE_X'],
                                text=df['text'],
                                mode='markers',
                                opacity=0.7,
                                marker={
                                    'size': 7,
                                    'color': df['colorScale'],
                                    'colorscale': 'Bluered',
                                    # 'line': {'width': 0.5, 'color': 'white'},
                                    'showscale': False
                                },
                                name='locations'
                            )
                        ],
                        'layout': go.Layout(
                            title='Geographic Distribution of 10,000 Sample Dispatches',
                            autosize=True,
                            height=600,
                            hovermode='closest',
                            mapbox={
                                'accesstoken': mapBoxKey,
                                'bearing': 0,
                                'center': {
                                    'lat': 39.12,
                                    'lon': -84.52
                                },
                                'pitch': 0,
                                'zoom': 10
                            }

                        )
                    }
                )
            ]),


            html.Div([
                dcc.Graph(
                    id='Box Plot',
                    figure={
                        'data': [
                            go.Box(
                                y=df[df['INCIDENT_TYPE_ID'] == i]['dispatch_to_arrival_time'],
                                name=i,
                                marker={
                                    'color': 'rgb(7,40,89)'
                                },
                                boxmean=True,
                                line={
                                    'color': 'rgb(7,40,89)'
                                },
                            ) for i in df.INCIDENT_TYPE_ID.unique()
                        ],
                        'layout': go.Layout(
                            title='Box Plots of Dispatch Times by Incident Type'
                        )
                    }
                ),
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                # Histogram of the selected plots, broken down into incident types
                dcc.Graph(
                    id='Histogram',
                    figure={
                        'data': [
                            go.Histogram(
                                histfunc='count',
                                x=df['INCIDENT_TYPE_ID'],
                                y=df['dispatch_to_arrival_time'],
                                name='Distribution of Incident Types'
                            )
                        ],
                        'layout': go.Layout(
                            title='Distribution of Incident Types'

                        )
                    }

                )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

                html.P(''),
                html.P(''),
                html.P(''),
                html.P('.'),

        html.Div([
            dcc.RadioItems(
                id='detailSelect',
                options=[
                    {'label': 'Incident Types', 'value': 'IT'},
                    {'label': 'Time of Day', 'value': 'TOD'},
                    {'label': 'Month', 'value': 'M'},
                    {'label': 'Day of the Week', 'value': 'DOW'},
                    {'label': 'Neighborhood', 'value': 'N'}
                ],
                value='IT',
                labelStyle={'display': 'inline-block'}

            )
        ], style={'width': '49%', 'float': 'center', 'display': 'inline-block'})
            ]),


####### TAB 2 ###########
    elif tab == 'tab-2':
        return html.Div(children=[
            # Quick graph of the features from the GBRT
            html.H2(
                'What Causes the Variance in 9-1-1 Police Response Times in Cincinnati?',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            dcc.Markdown('''
Disclaimer: This is an INITIAL assessment of the 9-1-1 Dispatch data set, and conclusions should not be made until \
consultation with subject matter experts.

Problem Statement:  9-1-1 response times for the Cincinnati Police Department (CPD) varied significantly between \
October 2014 and July 2018.  Even when eliminating known test and erroneous 9-1-1 dispatch calls, 9-1-1 \
dispatch-to-arrival (DTA) times had an IQR of 9 minutes (Q3: 12 min, Q1: 3 min, Median: 6 min).    Standard Deviation \
was even higher ( 31 min), which was heavily influenced by a significantly large right skew (Mean: 31 min, Max DTA \
time:1442 min).   

Goal:  Identify the features that cause higher than normal response times, thus allowing the CPD to identify an \
optimization policy that would decrease overall 9-1-1 response time.    

Methodology:  Based on known parameters exhibited in the provided dataset, employ a tree-based ensemble method \
to model 9-1-1 Dispatch-to-Arrival times.  A tree-based model is preferred due to its innate ability \
to identify feature importance.  The chosen ensemble method is Gradient Boosted Regression Trees \
(GBRT, or Gradient Boosting), which was implemented from the sci-kit learn library.  Once a model is created that can \
accurately predict DTA time, analyze the dominant features in the model for potential optimization opportunities.

GBRT Model: 
18 tests were run while manually tuning hyperparameters. 
The following hyperparameters were utilized in the most current iteration: 
* Number of Estimators (n_estimators): 4000
* Maximum leaves per node: 16 (limits each tree to no more than four splits)
* Minimum Samples per Split: 2
* Learning Rate: 0.01
* Loss Function:  Least Absolute Deviation (lad)
* Subsample: 0.8

Model Evaluation:
* Run Time: 6.52 Hours
* Explained Variance: 0.178
* MSE: 819.7286
* medAE (median Absolute Error): 3.20
* r^2: 0.1485
* MSLE (Mean Squared Log Error): 0.6301            

Summary:
While the median Absolute Error of 3.2 minutes is potentially acceptable, the delta between medAE and MSE (and the \
extremely high MSE) is indicative of an underfitted model with high outliers (i.e. the model is not fitting the \
extremely high dispatch times well).  It does outline some noticeable features however.  Geography (in the form of \
latitude and longitude) appears to be a dominant explanatory feature in the data set, leading us to conclude that \
location in Cincinnati plays a factor in police response times.  Several locations in the periphery of the city (and \
a few locations in the center) appear to have noticeably longer response times (see map below). Time of day also \
appears to be fairly prominent, with all times around 6am (presumably around shift change) having longer response times. 

The underfitting issue can be addressed through the following three courses of action:
1. Expand and refine the feature space.  Three improvements specifically could yield better results.  First, the \
geographic features (latitude and longitude) were given random noise in the data set to address privacy issues.  The \
original data could yield a more acurate model.  Second, I hypothesize that the quantity of simultaneous dispatches is \
correlated to DTA time.  Adding this feature set I believe would improve the model as well.  Third, further filtering \
is likely required in the data set to eliminate noise, which requires consulting with subject matter experts who are \
familiar with the data.
2. Select a more complex model.  Gradient boosting is known to have issues when fitting categorical feature spaces.  \
Furthermore, categorical features were addressed in this model using the one-hot method, which is perhaps not \
sophisticated enough for this data set.  Improved models include catBoost (a derivative of GBRT optimized for \
categorical feature spaces) or neural networks (CNN/RNN).  CatBoosting is a form of gradient boosting that can address \
categorical features more effectively, and if utilized would still provide an identification of feature importance.
3.  Employ a more sophisticated method for hyperparameter tuning.  Hyperparameters for this model were selected based \
on historic parameters that were known to be effective (i.e. low learning rates with high number of estimators, \
employing subsampling, etc).  Other parameters were identified as optimal given the skewed nature of the data \
(employing least absolute deviation as the loss function).  Future hyperparameter selection can be improved by \
utilizing grid searching or Bayesian optimization methods.              
            
            '''),


            html.Div([
                dcc.Graph(
                    id='FEATURES GRAPH',
                    figure={
                        'data': [
                            go.Bar(
                                x=[k for k in features],
                                y=[v for v in features.values()]
                            )
                        ],
                        'layout': go.Layout(
                            title='RELATIVE FEATURE IMPORTANCE*',
                            xaxis={'title': 'Features'},
                            yaxis={'title': 'Importance (100 = Most Significant Feature)'},
                        )
                    }

                ),
                html.P('*Only features with greater than 5% of the most important feature is visualized'),

                dcc.Graph(
                    id='example-graph-2',
                    figure={
                        'data': [
                            {'x': devianceAnalysis['numEstimators'], 'y': devianceAnalysis['TrainingSetDeviance'],
                             'type': 'line', 'name': 'Training Set Deviance'},
                            {'x': devianceAnalysis['numEstimators'], 'y': devianceAnalysis['TestSetDeviance'],
                             'type': 'line', 'name': 'Test Set Deviance'},
                        ],
                        'layout': go.Layout(
                            title='Deviance Analysis (Loss Function = Least Absolute Deviance)',
                            xaxis={'title': 'Number of Estimators'},
                            yaxis={'title': 'Deviance'},
                            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 1, 'y': 1}

                        )
                    }
                )
            ]),
            html.H2(
                'Explore:  Use the map below to compare how dispatch times vary by location and time',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),
            html.H3(
                'Data = 20,000 Random Dispatches.  Coloring is based on PREDICTED response time, not actual',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            html.Div(
                dcc.Graph(
                    id='Cincinnati 9-1-1 Calls: Model Analysis 1',
                    figure={
                        'data': [
                            go.Scattermapbox(
                                lon=modelAnalysis['Longitude'],
                                lat=modelAnalysis['Latitude'],
                                text=modelAnalysis['text'],
                                mode='markers',
                                opacity=0.7,
                                marker={
                                    'size': 7,
                                    'color': modelAnalysis['colorScale'],
                                    'colorscale': 'Bluered',
                                    # 'line': {'width': 0.5, 'color': 'white'},
                                    'showscale': False
                                },
                                name='locations'
                            )
                        ],
                        'layout': go.Layout(
                            autosize=True,
                            #width=1000,
                            height=700,
                            hovermode='closest',
                            mapbox={
                                'accesstoken': mapBoxKey,
                                'bearing': 0,
                                'center': {
                                    'lat': 39.12,
                                    'lon': -84.52
                                },
                                'pitch': 0,
                                'zoom': 10
                            }

                        )
                    }
                )
            ,style = {'width': '100%', 'float': 'center','display': 'inline-block'} ),


        ])
####### TAB 3 ###########
    elif tab == 'tab-3':
        return html.Div(children=[
            # Quick graph of the features from the GBRT
            html.H2(
                'What Causes Response Time Variability?  Can We Predict 9-1-1 Police Response Times in Cincinnati?',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            dcc.Markdown('''
Disclaimer: This is an INITIAL assessment of the 9-1-1 Dispatch data set, and conclusions should not be made until \
consultation with subject matter experts.

Problem Statement:  9-1-1 response times can often mean the difference between life or death.  How quickly police \
can respond to an assault in progress or a robbery, for example, can mean the difference between successfully \
stopping a crime in progress or preventing it from becoming more serious.  By using publicly available data \
(9-1-1 dispatches: https://data.cincinnati-oh.gov/Safer-Streets/PDI-Police-Data-Initiative-Police-Calls-for-Servic/gexm-h6bt), \
can we identify the features that cause variability in 9-1-1 responses and potentially predict them?  For example, \
are there higher-than-normal response times in some areas and at certain times of day? 

Goal:  Identify the features that cause higher than normal response times, thus allowing the CPD to identify an \
optimization policy that would improve 9-1-1 response times.      

Methodology:  Based on known parameters exhibited in the provided dataset, employ a tree-based ensemble method \
to model 9-1-1 Dispatch-to-Arrival times.  A tree-based model is preferred due to the suspected non-linear relationships \
with geo-locational data.  Tree-based models can also identify feature importance fairly easily, which will help us in \
policy determination.  The chosen ensemble method is a version of Gradient Boosted Regression Trees \
(GBRT, or Gradient Boosting), called CatBoost.  CatBoost is an open-source GBRT \
library developed by Yandex (https://tech.yandex.com/catboost/).  It is a significant improvement over earlier \
generation GBRT algorithms, and is particularly well suited for mixed numerical and categorical data.  

A bayesian optimization algorithm was implemented using the hyperopt library to find the most optimal hyperparameters \
for this data set.  100 cross-validated training sessions were done with the training set with the following \
hyperparameters given initial distributions:

iterations: 1000 (an over fitting check was implemented that will stop the training when detected)

learning_rate: Log uniform distribution between 0.005 and 0.2 (places initial emphasis in low learning rates)

loss_function: 'MAE' (The data has several very large outlier labels that I believe are erroneous, but will need to verify \
with subject matter experts.  Until then, I intend to use MAE over RMSE since MAE is more robust against outliers, \
at the expense of more compute resources and lower explanation for extremely large response times)

max_depth: Uniform integer distribution between 2 and 10 

bootstrap_type: bayesian (Not to be confused with the bayesian optimization of all hyperparameters - this is the \
method for sampling weights of objects(rows))

bagging_temperature: Uniform distribution between 0.0 and 100 (Defines the settings of the bayesian bootstrap.  The \
higher the weight, the more aggressive the bootstrapping)

random_strength: Uniform distribution between 0.0 and 100 (score the standard deviation of the model - helps avoid \
overfitting the model)

                   
                   '''),
            html.Div([
                dcc.Graph(
                    id='CROSS VALIDATION PLOT',
                    figure={
                        'data': catBoostCVData,
                        'layout': go.Layout(
                            title='CROSS VALIDATION PLOT',
                            # xaxis={'title': 'Features'},
                            # yaxis={'title': 'Importance'},
                            paper_bgcolor='rgb(255,255,255)',
                            plot_bgcolor='rgb(229,229,229)',
                            xaxis=dict(
                                title='Iterations',
                                gridcolor='rgb(255,255,255)',
                                range=[1, 1000],
                                showgrid=True,
                                showline=False,
                                showticklabels=True,
                                tickcolor='rgb(127,127,127)',
                                ticks='outside',
                                zeroline=False
                            ),
                            yaxis=dict(
                                title='Error (MAE)',
                                gridcolor='rgb(255,255,255)',
                                showgrid=True,
                                showline=False,
                                showticklabels=True,
                                tickcolor='rgb(127,127,127)',
                                ticks='outside',
                                zeroline=False
                            ),
                            margin=dict(
                                l=120,
                                r=10,
                                t=140,
                                b=80
                            ),
                        )
                    }

                ),
                html.P('Confidence bands are +/- 2 standard deviations following a 9-fold cross validation',
                       style={
                           'textAlign': 'center',
                           # 'color': colors['text']
                       }
                       ),
                dcc.Markdown('''
I ran 100 9-fold cross validations, initialized with the previous distributions.   Of the 100 cross validations,
the trial with the most optimal loss score (measured in Mean Average Error, or MAE) had the following parameters:

Number of iterations: 1000

bagging_temperature: 0.0254 (the bayesian optimization was gravitating more towards lower bagging)

learning_rate: 0.1985 

max_depth: 10.0

random_strength: 1.5111 (Random strength distribution was relatively bimodal, with concentrations in the low \
range (~1) and others much higher (40-50).  Random strength can help control overfitting, so if this is an issue \
with our model then we can look at exploring higher values)

The cross validation plot above shows the training and test set MAE for the aforementioned hyperparameters.   \
The 2-sigma standard deviations show the confidence bands for the 9 folds.  We start to \
see diminishing improvements in our model after ~500 iterations, but relatively good robustness against \
overfitting (standard deviation remains relatively constant).  Having an MAE score of ~6 is not ideal -  \
we can loosely interpret this as being ~6 minutes off on average on our prediction.  Future improvements \
to the model will likely involve more complex feature engineering, engagement with key stakeholders to filter erroneous \
data and identify new features, and exploring different linear and nonlinear models 

While I would like to see better results than ~6 MAE, I feel we can gain insightful information into our overall dataset by \
applying this model.  Thus using the hyperparameters above I trained the same model against 75% of the data.  The two \
graphs below provide insight into the features that are most important to the model as well as the top 50 interactions.           
                '''),

            ]),


            html.Div([
                dcc.Graph(
                    id='FEATURES GRAPH',
                    figure={
                        'data': [
                            go.Bar(
                                x=featureImportance.index ,#[k for k in featureImportance],
                                y=featureImportance.values #[v for v in featureImportance.values()]
                            )
                        ],
                        'layout': go.Layout(
                            title='RELATIVE FEATURE IMPORTANCE*',
                            xaxis={'title': 'Features'},
                            yaxis={'title': 'Importance'},
                        )
                    }

                ),

            ]),
            html.Div([
                dcc.Graph(
                    id='FEATURES INTERACTIONS GRAPH',
                    figure={
                        'data': [
                            go.Bar(
                                x=featureInteractions.values,  # [k for k in featureImportance],
                                y=featureInteractions.index,  # [v for v in featureImportance.values()]
                                orientation = 'h'
                            )
                        ],
                        'layout': go.Layout(
                            title='TOP 50 FEATURE INTERACTIONS',
                            height = 1000,
                            xaxis={'title': 'Feature Interactions'},
                            #yaxis={'title': 'Feature Interaction Pair'},
                            margin=dict(
                                l=230,
                                r=10,
                                t=140,
                                b=80
                            )
                        )
                    }

                ),

            ]),

            html.Div([
                dcc.Graph(
                    id='Cincinnati 9-1-1 Calls: Comparing Actual and Predicted Response Times',
                    figure={
                        'data': [
                            go.Scattermapbox(
                                lon=catBoost_df['Longitude'],
                                lat=catBoost_df['Latitude'],
                                text=catBoost_df['text'],
                                mode='markers',
                                opacity=0.7,
                                marker={
                                    'size': 7,
                                    'color': np.absolute(catBoost_df['color']),
                                    'colorscale': 'RdBu',
                                    # 'line': {'width': 0.5, 'color': 'white'},
                                    'showscale': False
                                },
                                name='locations'
                            )
                        ],
                        'layout': go.Layout(
                            title='Geographic Distribution of 10,000 Sample Dispatches',
                            autosize=True,
                            height=600,
                            hovermode='closest',
                            mapbox={
                                'accesstoken': mapBoxKey,
                                'bearing': 0,
                                'center': {
                                    'lat': 39.12,
                                    'lon': -84.52
                                },
                                'pitch': 0,
                                'zoom': 10
                            }

                        )
                    }
                ),
                dcc.Markdown('''The graphic above plots a sample of 10,000 9-1-1 dispatches.  The model has been \
applied to this sample dataset, with predicted response times visible when hovering over a data point.  Each point \
is colored based on accuracy of the prediction (i.e. red points indicate a bad prediction)




.
             '''),
            ]),





        ])



'''Map Callback function.  takes inputs from the year selector and incident type selector
    to filter down what is displayed on the map'''
@app.callback(
    dash.dependencies.Output('Cincinnati 9-1-1 Calls: Dispatch to Arrival Effectiveness', 'figure'),
    [dash.dependencies.Input('yearSelector', 'value'),
     dash.dependencies.Input('incident_type', 'value')])
def update_year(selected_year, incidents):

    if selected_year != 'All Years' and incidents == []:
        filtered_df = df[df.year == selected_year]
    elif selected_year != 'All Years' and incidents != []:
        filtered_df = df[df.year == selected_year]
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    elif selected_year == 'All Years' and incidents != []:
        filtered_df = df
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    else:
        filtered_df = df
    traces = []

    traces.append(go.Scattermapbox(
        lon=filtered_df['LONGITUDE_X'],
        lat=filtered_df['LATITUDE_X'],
        text=filtered_df['text'],
        mode='markers',
        opacity=0.7,
        marker={
            'size': 7,
            'color': filtered_df['colorScale'],
            'colorscale': 'Bluered',
            # 'line': {'width': 0.5, 'color': 'white'},
            'showscale': False
        },
        name='locations'
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title='Geographic Distribution of 10,000 Sample Dispatches',
            autosize=True,
            hovermode='closest',
            mapbox={
                'accesstoken': mapBoxKey,
                'bearing': 0,
                'center': {
                    'lat': 39.12,
                    'lon': -84.52
                },
                'pitch': 0,
                'zoom': 10
            }
            #xaxis={'title': 'Longitude'},
            #yaxis={'title': 'Latitude'},
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            #hovermode='closest'
        )
    }


'''Incident boxplot callback.  The incident type boxplot will only display the data selected on the
    map.'''
@app.callback(
    dash.dependencies.Output('Box Plot', 'figure'),
    [dash.dependencies.Input('yearSelector', 'value'),
     dash.dependencies.Input('incident_type', 'value'),
     dash.dependencies.Input('detailSelect', 'value')
     ]
    )
def update_incidents(selected_year, incidents,detail):
    #Create a temporary array of data points for the box plot
    #FUTURE WORK: Integrate the selections on the graph

    #If a year is selected but no incidents...
    if selected_year != 'All Years' and incidents == []:
        filtered_df = df[df.year == selected_year]
    #If a year and incidents are selected...
    elif selected_year != 'All Years' and incidents != []:
        filtered_df = df[df.year == selected_year]
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    #If no year is selected but an incident is selected...
    elif selected_year == 'All Years' and incidents != []:
        filtered_df = df
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    #Finally, if no year is selected and no incidents
    else:
        filtered_df = df

#Modify the figure.  Future work will require the detailSelect if statements here
    figureValues = details.loc[detail]
    if detail == 'IT':
        figureValues['xAxisValues'] = filtered_df.INCIDENT_TYPE_ID.unique()
    elif detail == 'N':
        figureValues['xAxisValues'] = filtered_df.NEIGHBORHOOD.unique()

    figure = {
        'data': [
            go.Box(
                y=filtered_df[filtered_df[figureValues.at['colNameInDF']] == i]['dispatch_to_arrival_time'],
                name=i,
                marker={
                    'color': 'rgb(7,40,89)'
                },
                boxmean=True,
                line={
                    'color': 'rgb(7,40,89)'
                },
            ) for i in figureValues.at['xAxisValues'] #filtered_df.INCIDENT_TYPE_ID.unique()
        ],
        'layout': go.Layout(
            title=  figureValues.at['bTitles'] #'Box Plots of Dispatch Times by Incident Type'
        )
    }
    return figure


'''Incident histogram callback.  The incident type histogram will only display the data selected on the
    map.'''
@app.callback(
    dash.dependencies.Output('Histogram', 'figure'),
    [dash.dependencies.Input('yearSelector', 'value'),
     dash.dependencies.Input('incident_type', 'value'),
     dash.dependencies.Input('detailSelect', 'value')
     ]
    )
def update_incidents(selected_year, incidents,detail):
    #Create a temporary array of data points for the box plot
    #FUTURE WORK: Integrate the selections on the graph

    #If a year is selected but no incidents...
    if selected_year != 'All Years' and incidents == []:
        filtered_df = df[df.year == selected_year]
    #If a year and incidents are selected...
    elif selected_year != 'All Years' and incidents != []:
        filtered_df = df[df.year == selected_year]
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    #If no year is selected but an incident is selected...
    elif selected_year == 'All Years' and incidents != []:
        filtered_df = df
        filtered_df = filtered_df.loc[filtered_df['INCIDENT_TYPE_ID'].isin(incidents)]
    #Finally, if no year is selected and no incidents
    else:
        filtered_df = df

    # Modify the figure.  Future work will require the detailSelect if statements here
    figureValues = details.loc[detail]
    if detail == 'IT':
        figureValues['xAxisValues'] = filtered_df.INCIDENT_TYPE_ID.unique()
    elif detail == 'N':
        figureValues['xAxisValues'] = filtered_df.NEIGHBORHOOD.unique()


    figure = {
                'data' : [
                    go.Histogram(
                        histfunc = 'count',
                        x = filtered_df[figureValues.at['colNameInDF']],
                        y = filtered_df['dispatch_to_arrival_time'],

                        name = figureValues.at['hTitles']
                    )
                ],
                'layout': go.Layout(
                    title = figureValues.at['hTitles']

                )
            }
    return figure




####################################################################################################################
####################################################################################################################
####################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)