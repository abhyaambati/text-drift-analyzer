import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from drift_analyzer import DriftAnalyzer
import base64
import numpy as np
from scipy import stats
import io
from sample_data_generator import create_sample_dataset

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def create_drift_timeline(df):
    fig = go.Figure()
    
    # Add main drift score line
    fig.add_trace(go.Scatter(
        x=df['timestamps'],
        y=df['drift_score'],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='#1f77b4')
    ))
    
    # Add statistical drift metrics
    fig.add_trace(go.Scatter(
        x=df['timestamps'],
        y=df['mean_drift'],
        mode='lines',
        name='Mean Drift',
        line=dict(color='#ff7f0e', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamps'],
        y=df['std_drift'],
        mode='lines',
        name='Std Drift',
        line=dict(color='#2ca02c', dash='dash')
    ))
    
    # Add anomaly points
    anomaly_mask = df['anomaly_scores'] > (df['anomaly_scores'].mean() + 2 * df['anomaly_scores'].std())
    fig.add_trace(go.Scatter(
        x=df.loc[anomaly_mask, 'timestamps'],
        y=df.loc[anomaly_mask, 'drift_score'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title='Drift Analysis Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Score',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_drift_distribution(df):
    fig = go.Figure()
    
    # Add drift score histogram
    fig.add_trace(go.Histogram(
        x=df['drift_score'],
        name='Drift Score',
        nbinsx=30,
        histnorm='probability'
    ))
    
    # Add KDE
    kde = stats.gaussian_kde(df['drift_score'])
    x_range = np.linspace(df['drift_score'].min(), df['drift_score'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde(x_range),
        name='KDE',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Distribution of Drift Scores',
        xaxis_title='Drift Score (%)',
        yaxis_title='Probability',
        template='plotly_white'
    )
    
    return fig

def create_metrics_correlation(df):
    metrics = ['drift_score', 'mean_drift', 'std_drift', 'kl_divergence']
    corr_matrix = df[metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Metrics Correlation Matrix',
        template='plotly_white'
    )
    
    return fig

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("LLM Response Drift Analyzer", className="text-center my-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Management", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='upload-data',
                                children=dbc.Button('Upload Data', color='primary'),
                                multiple=False
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Button('Load Sample Data', id='load-sample', color='success', className='ms-2'),
                        ], width=6)
                    ]),
                    html.Div(id='output-data-upload'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Export Analysis', id='export-analysis', color='info', className='mt-2'),
                            dcc.Download(id='download-analysis')
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Drift Timeline", className="card-title"),
                    dcc.Graph(id='drift-timeline')
                ])
            ])
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Drift Distribution", className="card-title"),
                    dcc.Graph(id='drift-distribution')
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Metrics Correlation", className="card-title"),
                    dcc.Graph(id='metrics-correlation')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Analysis Summary", className="card-title"),
                    html.Div(id='analysis-summary')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

@app.callback(
    [dash.Output('drift-timeline', 'figure'),
     dash.Output('drift-distribution', 'figure'),
     dash.Output('metrics-correlation', 'figure'),
     dash.Output('analysis-summary', 'children'),
     dash.Output('output-data-upload', 'children')],
    [dash.Input('upload-data', 'contents'),
     dash.Input('load-sample', 'n_clicks')]
)
def update_graphs(contents, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, {}, {}, "Please upload data or load sample data to begin analysis", ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'load-sample':
        # Generate sample dataset
        df = create_sample_dataset(n_samples=1000)
        message = f"Loaded sample dataset with {len(df)} samples"
    else:
        if contents is None:
            return {}, {}, {}, "Please upload data or load sample data to begin analysis", ""
        
        # Parse the uploaded data
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(pd.io.StringIO(decoded.decode('utf-8')))
        message = f"Uploaded dataset with {len(df)} samples"
    
    # Convert timestamp column to datetime
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Create visualizations
    timeline_fig = create_drift_timeline(df)
    distribution_fig = create_drift_distribution(df)
    correlation_fig = create_metrics_correlation(df)
    
    # Generate summary statistics
    analyzer = DriftAnalyzer()
    results = analyzer.analyze_drift(df['texts'].tolist(), df['timestamps'].tolist())
    report = analyzer.generate_report(results)
    
    summary = html.Div([
        html.H6("Key Metrics:"),
        dbc.Row([
            dbc.Col([
                html.P(f"Total Samples: {report['total_samples']}"),
                html.P(f"Average Drift: {report['average_drift']:.2f}%"),
                html.P(f"Maximum Drift: {report['max_drift']:.2f}%")
            ], width=4),
            dbc.Col([
                html.P(f"Drift Trend: {'Increasing' if report['drift_trend'] > 0 else 'Decreasing'}"),
                html.P(f"Significant Drift Events: {report['significant_drift_events']}"),
                html.P(f"Anomaly Count: {report['anomaly_count']}")
            ], width=4),
            dbc.Col([
                html.P(f"Drift Volatility: {report['drift_volatility']:.2f}%"),
                html.P(f"Mean Drift Trend: {'Increasing' if report['mean_drift_trend'] > 0 else 'Decreasing'}"),
                html.P(f"KL Divergence Trend: {'Increasing' if report['kl_divergence_trend'] > 0 else 'Decreasing'}")
            ], width=4)
        ])
    ])
    
    return timeline_fig, distribution_fig, correlation_fig, summary, message

@app.callback(
    dash.Output('download-analysis', 'data'),
    dash.Input('export-analysis', 'n_clicks'),
    [dash.State('upload-data', 'contents'),
     dash.State('load-sample', 'n_clicks')]
)
def export_analysis(n_clicks, contents, sample_clicks):
    if not n_clicks:
        return None
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    # Get the current data
    if sample_clicks:
        df = create_sample_dataset(n_samples=1000)
    else:
        if contents is None:
            return None
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(pd.io.StringIO(decoded.decode('utf-8')))
    
    # Convert timestamp column to datetime
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Generate analysis
    analyzer = DriftAnalyzer()
    results = analyzer.analyze_drift(df['texts'].tolist(), df['timestamps'].tolist())
    report = analyzer.generate_report(results)
    
    # Create export data
    export_data = {
        'summary': report,
        'drift_scores': results['drift_score'],
        'timestamps': results['timestamps'],
        'anomaly_scores': results['anomaly_scores']
    }
    
    # Convert to DataFrame for export
    export_df = pd.DataFrame(export_data)
    
    # Create the export file
    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    
    return dict(
        content=buffer.getvalue(),
        filename='drift_analysis_export.csv',
        type='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True) 