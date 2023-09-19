import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import os
import numpy as np


load_figure_template("CYBORG")
DATA_DIR = "dataset"
MINS_DIR = "5-min-2-hours-0.5-overlap-normal-beat-with-ectopic-info"

def load_data(dataset="hrv"):
    df_nsrdb = pd.read_csv(os.path.join(DATA_DIR, MINS_DIR, "df_nsrdb.csv"))
    df_chfdb = pd.read_csv(os.path.join(DATA_DIR, MINS_DIR, "df_chfdb.csv"))
    df_nsr2db = pd.read_csv(os.path.join(DATA_DIR, MINS_DIR, "df_nsr2db.csv"))
    df_chf2db = pd.read_csv(os.path.join(DATA_DIR, MINS_DIR, "df_chf2db.csv"))
    df = pd.concat([df_nsrdb, df_chfdb, df_nsr2db, df_chf2db], axis=0).reset_index(drop=True)
    
    map_gender = {"m": "Male", "f": "Female"}
    df = df[(df.age != "?") & (df.gender != "?")]
    df = df.replace(["?"], np.nan)
    df['age'] = np.array(df['age'], dtype=np.float16)
    df['gender'] = df['gender'].map(lambda x: map_gender[x.lower()])
    df["record_ids"] = df["id"].map(lambda x: "-".join(x.split("-")[:2]))
    df = df[["id", "record_ids"] + df.drop(columns=["id", "record_ids"]).columns.to_list()]
    df = df.drop(columns=['tinn'])
    df = df.replace([np.inf, -np.inf], np.nan)
    # df = df[df['age'] >= 10] # useless since no data with age > 10 in CHFDB, CHF2DB, NSRDB, NSR2DB
    df = df[df["normal_beats_ratio"] >= 0.9]

    if 'db_source' in df.columns:
        db_source = df['db_source']
        # df = df.drop(columns=['db_source'])
    drop_columns = ["signal_length", "recording_time_hours", "recording_time_seconds"]
    df = df.drop(columns=drop_columns)

    df = df.dropna()
    df["ratio_sd1_sd2"] = 1 / df["ratio_sd2_sd1"]
    
    map_risk = {1: "CHF", 0: "Healthy"}
    df["risk"] = df["risk"].map(lambda x: map_risk[x])
    
    train_record_ids = pd.read_csv(os.path.join(DATA_DIR, "train_record_ids_7-3_post-filter.csv"))
    test_record_ids = pd.read_csv(os.path.join(DATA_DIR, "test_record_ids_7-3_post-filter.csv"))
    df_train = df[
        df["record_ids"].isin(train_record_ids["id"].values)
    ]
    df_test = df[
        df["record_ids"].isin(test_record_ids["id"].values)
    ]
    
    data_info = ['age', 'gender', 'db_source']
    time_domain_indices = ['mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr']
    freq_domain_indices = ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf']
    geometrical_indices = ['triangular_index']
    poincare_plot_indices = ['sd1', 'sd2', 'ratio_sd1_sd2']
    csi_csv_indices = ['csi', 'cvi', 'Modified_csi']
    sampen = ['sampen']
    disease_info = ['cardiac_info', 'risk']
    
    columns_needed = data_info + \
                    time_domain_indices + freq_domain_indices + \
                    poincare_plot_indices + \
                    csi_csv_indices + \
                    disease_info
                    
    df_train = df_train[columns_needed]
    df_test = df_test[columns_needed]

    return df_train


df = load_data()
hrv_indices = df.drop(columns=["risk", "age", "gender", "db_source", "cardiac_info"]).columns

## Charts
def create_bar_chart_gender(legend_column="db_source"):
    df_gender_counts = df[['gender', 'db_source', 'risk']].value_counts().reset_index()
    # df_gender_counts[[legend_column, "count"]].groupby(by=legend_column).sum()
    # if legend_column == "db_source":
    #     df.groupby(legend_column)
    bar_fig = px.bar(
        df_gender_counts, x='gender', y='count', color=legend_column,
        title="Gender distribution",
        template="plotly_dark"
    )
    bar_fig.update_layout(height=600, )
    return bar_fig

def create_histogram_chart(legend_column="db_source"):
    hist_fig = px.histogram(
        df, x="age", color=legend_column,
        template="plotly_dark",
        title="Histogram distribution of age"
    )
    hist_fig.update_layout(height=600)
    return hist_fig

def create_box_chart_age(display_points=None, show_gender=False):
    box_fig = px.box(
        df, x=["db_source"], y="age", color='gender' if show_gender else None,
        template="plotly_dark",
        points="all" if display_points else None,
        title="Box plots of Age <br>based on each database"
    )
    box_fig.update_layout(height=600)
    return box_fig

def create_scatter_chart(x_axis="age", y_axis="rmssd", color_encode=False, symbol_encode=False):
    scatter_fig = px.scatter(
        df, x=x_axis, y=y_axis, 
        color="risk" if color_encode else None, 
        symbol="gender" if symbol_encode else None,
        title="{} vs {}".format(x_axis.upper(), y_axis.upper()),
        template="plotly_dark"
    )
    scatter_fig.update_layout(height=600)
    return scatter_fig
    
df_avg_risk = df[list(hrv_indices) + ["risk"]].groupby("risk").mean().reset_index()
df_avg_db_source = df[list(hrv_indices) + ["db_source"]].groupby("db_source").mean().reset_index()
df_avg_gender = df[list(hrv_indices) + ["gender"]].groupby("gender").mean().reset_index()
df_avg_cardiac_info = df[list(hrv_indices) + ["cardiac_info"]].groupby("cardiac_info").mean().reset_index()
def create_bar_chart(x="risk", hrv_index="rmssd"):
    if x == "risk":
        df = df_avg_risk
    elif x == "db_source":
        df = df_avg_db_source
    elif x == "gender":
        df = df_avg_gender
    elif x == "cardiac_info":
        df = df_avg_cardiac_info
    bar_fig = px.bar(
        df, x=x, y=hrv_index,
        title="Average of HRV Index: {} <br>between {}".format(hrv_index.upper(), x.upper()),
        template="plotly_dark"
    )
    bar_fig.update_layout(height=600, )
    return bar_fig

def create_box_chart(x=["db_source"], y="rmssd", display_points=None):
    box_fig = px.box(
        df, x=x, y=y,
        template="plotly_dark",
        points="all" if display_points else None,
        title="Box plots of HRV Index <br>based on selected criteria"
    )
    box_fig.update_layout(height=600)
    return box_fig

## Widgets
# legend for gender distribution plot
bar_legend = dcc.Dropdown(id="bar_legend", options=["db_source", "risk", None], value="db_source", style={"display": "inline-block", "width": "50%"})

# Histogram plot of Age
hist_legend = dcc.Dropdown(id="hist_legend", options=["db_source", "gender", "risk"], value="db_source", style={"display": "inline-block", "width": "50%"})

# Box plot of Age
display_points_box_age = dcc.Checklist(id="display_points_box_age", options=["Display Points", ], style={"display": "inline-block", "width": "auto"})
show_gender = dcc.Checklist(id="show_gender", options=["Show Gender", ], style={"display": "inline-block", "width": "auto", "margin-left": "30px"})

# Scatter plot of Age
x_axis = dcc.Dropdown(id="x_axis", options=["age"] + list(hrv_indices), value="age", clearable=False, style={"display": "inline-block", "width": "50%"})
y_axis = dcc.Dropdown(id="y_axis", options=hrv_indices, value="rmssd", clearable=False, style={"display": "inline-block", "width": "50%"})
color_encode = dcc.Checklist(id="color_encode", options=["Risk", ], style={"display": "inline-block", "width": "auto"})
symbol_encode = dcc.Checklist(id="symbol_encode", options=["Gender", ], style={"display": "inline-block", "width": "70%", "margin-left": "30px"})

# Bar plot of HRV indices for average
x_axis_hrv_bar = dcc.Dropdown(id="x_axis_hrv_bar", options=["db_source", "gender", "risk", "cardiac_info"], value="risk", clearable=False, multi=False, style={"display": "inline-block", "width": "50%"})
y_axis_hrv_bar = dcc.Dropdown(id="y_axis_hrv_bar", options=hrv_indices, value="rmssd", clearable=False, multi=False, style={"display": "inline-block", "width": "50%"})

# Box plot of each database
x_axis_db = dcc.Checklist(id="x_axis_db", options=["db_source", "gender", "risk"], value=["db_source"], inline=True, inputStyle={"margin-left": "20px", "margin-right": "5px", "display": "40px"})
y_axis_hrv = dcc.Dropdown(id="y_axis_hrv", options=hrv_indices, value="rmssd", clearable=False, style={"display": "40px"})
display_points = dcc.Checklist(id="display_points", options=["Display Points", ], style={"display": "inline-block", "width": "auto"})


# Web App
app = Dash(
    title="HRV Dashboard",
    external_stylesheets=[dbc.themes.CYBORG]
)

app.layout = html.Div(
    children=[
        html.H1("HRV Analysis", style={"text-align": "center"}),
        html.Div("Explore relationship between various HRV indices of Healthy and Congestive Heart Failure (CHF) subjects", style={"text-align": "center"}),
        html.Br(),
        html.Br(),
        html.Div(
            children=[
                bar_legend,
                dcc.Graph(id="bar_chart_gender", figure=create_bar_chart_gender())
            ],
            style={"display": "inline-block", "width": "50%", "padding": "10px"}
        ),
        html.Div(
            children=[
                hist_legend,
                dcc.Graph(id="histogram_chart", figure=create_histogram_chart()),
            ],
            style={"display": "inline-block", "width": "50%", "padding": "10px"}
        ),
        html.Div(
            children=[
                display_points_box_age,
                show_gender,
                dcc.Graph(id="box_chart_age", figure=create_box_chart_age())
            ],
            style={"display": "inline-block", "width": "100%", "padding": "30px"}
        ),
        html.Div(
            children=[
                color_encode, symbol_encode,
                x_axis, 
                y_axis, 
                dcc.Graph(id="scatter_chart", figure=create_scatter_chart())
            ],
            style={"display": "inline-block", "width": "55%", "padding": "30px"}
        ),
        html.Div(
            children=[
                x_axis_hrv_bar,
                y_axis_hrv_bar,
                dcc.Graph(id="bar_chart", figure=create_bar_chart())
            ],
            style={"display": "inline-block", "width": "40%"}
        ),
        html.Br(),
        html.Div(
            children=[
                # html.P("X-Axis:", style={"display": "inline-block"}),
                x_axis_db, 
                # html.P("Y-Axis:"),
                y_axis_hrv,
                display_points,
                dcc.Graph(id="box_chart", figure=create_box_chart()),
            ],
            style={"display": "inline-block", "width": "100%", "padding": "30px"}
        ),
    ],
    style={"padding": "50px"}
)


## Callbacks
@callback(
    Output(component_id="bar_chart_gender", component_property="figure"),
    [
        Input(component_id="bar_legend", component_property="value")
    ]
)
def update_bar_chart_gender(legend_column):
    return create_bar_chart_gender(legend_column)

@callback(
    Output(component_id="histogram_chart", component_property="figure"),
    [
        Input(component_id="hist_legend", component_property="value")
    ]
)
def update_histogram_chart(legend_column):
    return create_histogram_chart(legend_column)

@callback(
    Output(component_id="box_chart_age", component_property="figure"),
    [
        Input(component_id="display_points_box_age", component_property="value"),
        Input(component_id="show_gender", component_property="value"),
    ]
)
def update_box_chart_age(display_points, show_gender):
    return create_box_chart_age(display_points, show_gender)

@callback(
    Output(component_id="scatter_chart", component_property="figure"), 
    [
        Input(component_id="x_axis", component_property="value"), 
        Input(component_id="y_axis", component_property="value"), 
        Input(component_id="color_encode", component_property="value"),
        Input(component_id="symbol_encode", component_property="value"),
    ]
)
def update_scatter_chart(x_axis, y_axis, color_encode, symbol_encode):
    return create_scatter_chart(x_axis, y_axis, color_encode, symbol_encode)

@callback(
    Output(component_id="bar_chart", component_property="figure"),
    [
        Input(component_id="x_axis_hrv_bar", component_property="value"),
        Input(component_id="y_axis_hrv_bar", component_property="value"),
    ]
)
def update_bar_chart(x, hrv_index):
    return create_bar_chart(x, hrv_index)

@callback(
    Output(component_id="box_chart", component_property="figure"),
    [
        Input(component_id="x_axis_db", component_property="value"),
        Input(component_id="y_axis_hrv", component_property="value"),
        Input(component_id="display_points", component_property="value"),
    ]
)
def update_box_chart(x, y, display_points):
    return create_box_chart(x, y, display_points)

server = app.server
if __name__ == "__main__":
    # app.run_server(debug=True, port=8051)
    app.run_server(debug=True)