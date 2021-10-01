# Standard library
import base64
import datetime
import io
import time
import os

# Dash
from dash import Dash, callback, Input, Output, State, callback_context, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Other
import numpy as np
import plotly.express as px
import pandas as pd

# Custom libraries
from src.search import search
from src.scrape_web import scrape_web_main
from src.nmf import nmf_main
from src.modules import mod_topic_by_domain

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__,
           external_stylesheets=[dbc.themes.SOLAR],
           suppress_callback_exceptions=True)



def generate_summary_text(keywords, num_topics):
    # keywords
    keyword_list = keywords.split(' ')
    keyword_list = [keyword for keyword in keyword_list if keyword.lower() not in ('and', 'or')]

    # Total queried documents
    total_queried = len(pd.read_csv('data\\search_results.csv'))

    # Total scraped documents
    scraped_text = pd.read_csv('data\\text_preprocessed.csv')
    scraped_text.text = scraped_text.text.fillna('')
    total_scraped = len(scraped_text)-sum(scraped_text.text == '')

    # Num topics generated
    num_topics = len(pd.read_csv('data\\nmf_output.csv').columns)

    # Most most represented topic(s)
    topic_per_sentence = pd.read_csv('data\\sentence_by_topic.csv')
    unique_topic_counts = np.unique(topic_per_sentence.topic, return_counts=True)
    most_refs_idx = np.where(np.max(unique_topic_counts[1]))[0]
    most_common_topics = unique_topic_counts[0][most_refs_idx]

    return [html.Code('{0}'.format(20)),
            ' pages were attempted to be scraped for each URL with keyword(s) ',
            html.Code('<{0}>'.format(','.join(keyword_list))),
            ', resulting in ',
            html.Code('{0}'.format(total_queried)),
            ' total links discovered and ',
            html.Code('{0}'.format(total_scraped)),
            ' total documents scraped.',
            html.Br(),
            html.Br(),
            html.Code('{0}'.format(num_topics)),
            ' topics were generated. The most represented topic(s) is (are) ',
            html.Code('<{0}>'.format(','.join(most_common_topics.astype(str)))),
            '.']



def generate_table(dataframe):
    contents = [dcc.Checklist(id='topic-checklist',
                                     options=[{'label':'({0})...{1}'.format(i, ' | '.join(dataframe[col].values)),
                                               'value':'{0}'.format(i)} for i, col in enumerate(dataframe.columns)],
                                     labelStyle={'display':'block'})]
    return contents

@app.callback(
    Output('display-num-topics', 'children'),
    [Input('topic-specificity-slider', 'drag_value')]
)
def update_slider_value(slider_value):
    return 'Choose topic specificity (# topics): {0}'.format(slider_value)

@app.callback(
    [Output('invalid-url-alert', 'children'),
     Output('loading-1', 'children'),
     Output('text-summary', 'children'),
     Output('grid-top-left', 'children'),
     Output('grid-top-right', 'children'),
     Output('specificity-slider', 'style')],
    [Input('run-nmf', 'n_clicks'),
     Input('topic-specificity-slider', 'value')],
    [State('keyword-input', 'value'),
     State('upload-urls', 'contents')]
)
def trigger_nmf(n_clicks, slider_val, keyword_input, domain_urls):
    trigger_id = callback_context.triggered[0]['prop_id']
    if (n_clicks is None and slider_val is None):
        raise PreventUpdate
    invalid_url_alert_component = []
    if trigger_id == 'run-nmf.n_clicks':
        # Read in URLs
        content_type, content_string = domain_urls.split(',')
        decoded = base64.b64decode(content_string)
        domain_urls = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        invalid_urls = search(keyword_input, domain_urls) # save invalid urls if exist
        invalid_url_alert_component = []
        invalid_urls = []
        if len(invalid_urls) > 0:
            alert_message = []
            for url in invalid_urls:
                alert_message += [url, html.Br()]
            alert_message = ["URLs omitted due to no search results", html.Br()] + alert_message[:-1]
            invalid_url_alert_component = [dbc.Alert(alert_message,
                                          color='danger',
                                          dismissable=True)]

        scrape_web_main('data\\search_results.csv')
        
        nmf_output = nmf_main('data\\text_preprocessed.csv', 5)
        slider_val = 5
    else:
        nmf_output = nmf_main('data\\text_preprocessed.csv', first_call=False, n_topics=slider_val)
    output = {}
    for topic in nmf_output:
        output[topic[0]] = topic[1]
    pd.DataFrame(output).to_csv('data\\nmf_output.csv', index=False)
    keyword_table = generate_table(pd.DataFrame(output))
    summary_text = generate_summary_text(keyword_input, slider_val)
    topic_by_domain_components = [dcc.Dropdown(id='topic-by-domain-dropdown',
                                             options=[{'label':'{0}'.format(i), 'value':'{0}'.format(i)} for i in range(len(output))],
                                             placeholder="Select a topic",
                                             clearable=False,
                                             searchable=False,
                                             style={'width':'55%'}),
                                dash_table.DataTable(id='topic-by-domain-table',
                                                     columns=[{'name': name, 'id': id} for name, id in zip(['Name', '% pages'], ['domain_name', '% pages'])],
                                                     data= [{'Name':['']},{'% pages':['']}],
                                                     page_size=10,
                                                     style_data={'whiteSpace': 'normal',
                                                                 'height': 'auto',
                                                                 'lineHeight': '15px'},
                                                     style_table={'width':'100%',
                                                                  'overflowX': 'auto',
                                                                  'textColor':'white'},
                                                     style_header={'backgroundColor':'#002a36'},
                                                     style_cell={'backgroundColor': '#01475a'})]
    invalid_url_alert_component = []
    return (invalid_url_alert_component,
            keyword_table,
            summary_text,
            dash_table.DataTable(id='raw_table',
                                 columns=[{'name': name, 'id': id} for name, id in zip(['Name', 'sentence', 'url'], ['domain_name', 'sentence', 'url'])],
                                 data= [{'Name':['']},{'sentence':['']},{'url':['']}],
                                 page_size=3,
                                 style_data={'whiteSpace': 'normal',
                                             'height': 'auto',
                                             'lineHeight': '15px'},
                                 style_table={'width':'100%',
                                              'overflowX': 'auto'},
                                 style_header={'backgroundColor':'#002a36'},
                                 style_cell={'backgroundColor': '#01475a'},
                                 export_format="csv"),
             topic_by_domain_components,
             {'display':'block'})

@app.callback(
    Output('topic-by-domain-table', 'data'),
    [Input('topic-by-domain-dropdown', 'value')]
)
def update_topic_by_domain(topic):
    if topic is None:
        raise PreventUpdate
    df = mod_topic_by_domain(topic)
    return df.to_dict('records')

@app.callback(
    Output('raw_table', 'data'),
    [Input('topic-checklist', 'value')]
)
def filter_table(checked_values):
    if checked_values is None:
        return [{'Name':None,'sentence':None,'url':None}]
    else:
        sentence_df = pd.read_csv('data\\sentence_by_topic.csv')
        sentence_df = sentence_df.loc[np.isin(sentence_df.topic, [int(val) for val in checked_values]), ['domain_name', 'sentence', 'url']]
        return sentence_df.to_dict('records')

@app.callback(
    Output('input-type', 'children'),
    [Input('query-or-import', 'value')]
)
def choose_input(button_val):
        return


app.layout = html.Div(children=[
                                html.H1("Topic modeler"),
                                html.Div(id='input-type', children=[html.Div(dcc.Input(id='keyword-input',
                                                                                        type='text',
                                                                                        debounce=False,
                                                                                        placeholder='Enter search query'),
                                                                             style={'display':'inline-block'}),
                                                                    html.Div(dcc.Upload(id='upload-urls',
                                                                                        children=html.Button('Upload URLs'),
                                                                                        multiple=False),
                                                                             style={'display':'inline-block'}),
                                                                    html.Div(html.Button('Run', id='run-nmf'),
                                                                             style={'display':'inline-block'})],
                                         style={'marginBottom':'5%'}),
                                html.Div(id='invalid-url-alert', children=[]),
                                html.Div(id='checklist-box',
                                         children= dcc.Loading(id = 'loading-1',
                                                               type='default',
                                                               fullscreen=False,
                                                               children = [dcc.Checklist(id='topic-checklist')]),
                                         style= {'display': 'inline-block',
                                                 'textAlign': 'left',
                                                 'height': '300px',
                                                 'width': '50%',
                                                 'overflowY': 'auto'}),
                                 html.Div(id='text-summary-box',
                                          children=[html.P(id='text-summary',
                                                           children=[],
                                                           style={'fontSize':'20px',
                                                                  'marginTop':'1rem',
                                                                  'marginBottom':'1rem',
                                                                  'marginLeft':'4rem'})],
                                          style={'display':'inline-block',
                                                 'textAlign':'left',
                                                 'height': '300px',
                                                 'width':'50%',
                                                 'verticalAlign':'top'}),
                                 html.Div(id='specificity-slider',
                                          children=[html.P(id='display-num-topics',
                                                           children=['Choose topic specificity (# topics): None']),
                                                    dcc.Slider(id='topic-specificity-slider',
                                                    min=1,
                                                    max=40,
                                                    step=1,
                                                    value=None,
                                                    updatemode='mouseup')],
                                          style={'display':'none'}),
                                 html.Div(id='grid-top-left',
                                          children=[],
                                          style={'width':'45%',
                                                 'display': 'inline-block',
                                                 'verticalAlign':'top',
                                                 'marginRight':'10%'}),
                                 html.Div(id='grid-top-right',
                                          children=[],
                                          style={'width':'45%',
                                                 'display': 'inline-block',
                                                 'verticalAlign':'top'}),
                                 html.Div(id='grid-bot-left',
                                          children=[],
                                          style={'width':'45%',
                                                 'display': 'inline-block',
                                                 'verticalAlign':'top',
                                                 'marginRight':'10%'}),
                                 html.Div(id='grid-bot-right',
                                          children=[],
                                          style={'width':'45%',
                                                 'display': 'inline-block',
                                                 'verticalAlign':'top'})],
                     style={'textAlign': 'center',
                            'alignItems': 'center',
                            'marginLeft':'20%',
                            'marginRight':'20%'})


if __name__ == '__main__':
    app.run_server(debug=False)
