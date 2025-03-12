import os
import json
from threading import Thread
import time
from typing import List, Dict, Any
import sys

import redis
import pandas as pd
import jsonpickle
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    MATCH,
    callback_context,
)
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from core.model import *
from core.system import *

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global variables
current_alerts = []
expanded_alerts = set()
expanded_recommendations = set()
connector_probs = {}

# Redis setup
r = redis.StrictRedis(host="localhost", port=6379)
pubsub = r.pubsub()
pubsub.subscribe(["breakage_probs", "recommendations"])


def write_file(s):
    """
    Takes a System and writes it to an array of System representing different ticks.

    Args:
        s: The System object to be written.
    """
    a = []

    fname = None  # default name
    if len(sys.argv) == 3:
        fname = sys.argv[2]

    if os.path.isfile(fname):
        with open(fname, "r") as f:
            a = jsonpickle.decode(f.read())
            a.append(s)
    with open(fname, "w") as f:
        f.write(jsonpickle.encode(a))


def get_color_from_probability(prob: float) -> str:
    """
    Converts a probability to a color gradient from green to red.

    Args:
        prob: The probability value (0-100).

    Returns:
        A string representing the RGB color.
    """
    red = int((prob / 100) * 255)
    green = int((1 - (prob / 100)) * 255)
    return f"rgb({red},{green},0)"


def create_probability_chart(df, title):
    """
    Creates a bar chart of probabilities for connectors.

    Args:
        df (pd.DataFrame): DataFrame containing connector probabilities.
        title (str): Title for the chart.

    Returns:
        go.Figure: A plotly Figure object representing the probability chart.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {title.split()[0]} detected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
    else:
        fig = go.Figure()
        for _, row in df.iterrows():
            color = get_color_from_probability(row["Probability"])
            fig.add_trace(
                go.Bar(
                    x=[row["Connector"]],
                    y=[row["Probability"]],
                    name=row["Connector"],
                    marker_color=color,
                    text=f"{row['Probability']:.2f}%",
                    textposition="outside",
                )
            )

    fig.update_layout(
        title=dict(
            text=title, font=dict(size=18, color="#333"), x=0.5, xanchor="center"
        ),
        xaxis_title="Connector",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(tickangle=-45),
        showlegend=False,
        height=400,
    )

    return fig


def create_alert_box(alert: Dict[str, Any], index: int) -> html.Div:
    """
    Creates a Dash component representing an alert box.

    Args:
        alert (Dict[str, Any]): Dictionary containing alert information.
        index (int): Index of the alert.

    Returns:
        html.Div: A Dash component representing the alert box.
    """
    alert_id = f"alert-{alert['created_at']}"
    is_alert_expanded = alert_id in expanded_alerts
    is_recommendation_expanded = alert_id in expanded_recommendations

    if alert["type"] == "high":
        alert_color = "danger"
        alert_icon = DashIconify(icon="mdi:alert", color="#dc3545", width=24, height=24)
    else:
        alert_color = "warning"
        alert_icon = DashIconify(
            icon="mdi:alert-outline", color="#ffc107", width=24, height=24
        )

    # Generate specific recommendations
    connector_num = int(alert["connector"][-1])
    recommendations = [
        f"1. Close valve VL{connector_num + 1} to isolate the affected area",
        f"2. Open connector PP{connector_num} to reroute flow",
        f"3. Dispatch maintenance team to inspect {alert['connector']}",
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    dbc.Button(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(alert_icon, width="auto", className="me-2"),
                                    dbc.Col(alert["title"], className="text-truncate"),
                                    dbc.Col(
                                        DashIconify(
                                            icon="mdi:chevron-down",
                                            color="#ffffff",
                                            width=24,
                                            height=24,
                                            className="float-end",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                align="center",
                                className="g-0",
                            ),
                        ],
                        id={"type": "alert-toggle", "index": alert_id},
                        color=alert_color,
                        className="text-left w-100 p-2",
                    )
                ],
                className=f"bg-{alert_color} text-white",
            ),
            dbc.Collapse(
                dbc.CardBody(
                    [
                        html.P(alert["content"], className="mb-3"),
                        dbc.Button(
                            [
                                DashIconify(
                                    icon="mdi:lightbulb-on-outline", className="me-2"
                                ),
                                "View Recommendation",
                            ],
                            id={"type": "recommendation-toggle", "index": alert_id},
                            color=alert_color,
                            outline=True,
                            size="sm",
                            className="mb-2",
                        ),
                        dbc.Collapse(
                            html.Div(
                                [
                                    html.Hr(),
                                    html.P(
                                        [
                                            DashIconify(
                                                icon="mdi:clipboard-text-outline",
                                                className="me-2",
                                            ),
                                            "Recommended Actions:",
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Ul(
                                        [html.Li(rec) for rec in recommendations],
                                        className="mb-3",
                                    ),
                                    html.P(
                                        [
                                            DashIconify(
                                                icon="mdi:robot-outline",
                                                className="me-2",
                                            ),
                                            f"Model used: {alert.get('model', 'N/A')}",
                                        ],
                                        className="text-muted small",
                                    ),
                                ]
                            ),
                            id={"type": "recommendation-collapse", "index": alert_id},
                            is_open=is_recommendation_expanded,
                        ),
                    ]
                ),
                id={"type": "alert-collapse", "index": alert_id},
                is_open=is_alert_expanded,
            ),
        ],
        className="mb-3 alert-card",
    )


# Layout components
system_recommendations = dbc.Card(
    [
        dbc.CardHeader("System Recommendations and Alerts"),
        dbc.CardBody(html.Div(id="system-recommendations-content")),
    ],
    className="mb-4",
)

app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.NavbarBrand("PF GUI", className="mx-auto"),
                                className="text-center",
                                width={"size": 6, "offset": 3},
                            ),
                            dbc.Col(
                                html.Div(id="current-time", className="text-light"),
                                width=3,
                                className="text-end",
                            ),
                        ],
                        className="w-100",
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Probability Charts"),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="tap-probabilities-chart"
                                                        ),
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="pipe-probabilities-chart"
                                                        ),
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="valve-probabilities-chart"
                                                        ),
                                                        md=4,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            ),
                            className="mb-4",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("System Alerts"),
                                        dbc.CardBody(
                                            html.Div(id="system-alerts-content")
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("System Overview"),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="system-explanation"),
                                                html.Hr(),
                                                html.Div(id="dynamic-insights"),
                                            ]
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
            ],
            fluid=True,
        ),
    ]
)

from datetime import datetime


@app.callback(
    Output("current-time", "children"), Input("interval-component", "n_intervals")
)
def update_time(n):
    """
    Updates the current time display.

    Args:
        n (int): Number of intervals (unused, but required for the callback).

    Returns:
        html.Span: A Dash component displaying the current time.
    """
    return html.Span(
        ["Current Time: ", html.Strong(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
    )


@app.callback(
    [
        Output("tap-probabilities-chart", "figure"),
        Output("pipe-probabilities-chart", "figure"),
        Output("valve-probabilities-chart", "figure"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_breakage_probabilities(n_intervals):
    """
    Updates the breakage probability charts for taps, pipes, and valves.

    Args:
        n_intervals (int): Number of intervals (unused, but required for the callback).

    Returns:
        tuple: Three plotly Figure objects for tap, pipe, and valve probability charts.
    """
    global connector_probs
    df = pd.DataFrame(
        list(connector_probs.items()), columns=["Connector", "Probability"]
    )
    df["Probability"] = df["Probability"].astype(float) * 100

    taps_df = df[df["Connector"].str.startswith("TP")]
    pipes_df = df[df["Connector"].str.match(r"^(PP|RP)")]
    valves_df = df[df["Connector"].str.startswith("VL")]

    tap_fig = create_probability_chart(taps_df, "Tap Breakage Probabilities")
    pipe_fig = create_probability_chart(pipes_df, "Pipe Breakage Probabilities")
    valve_fig = create_probability_chart(valves_df, "Valve Breakage Probabilities")

    return tap_fig, pipe_fig, valve_fig


@app.callback(
    Output("system-recommendations-content", "children"),
    Input("interval-component", "n_intervals"),
)
def update_system_recommendations(n_intervals):
    """
    Updates the system recommendations and alerts.

    Args:
        n_intervals (int): Number of intervals (unused, but required for the callback).

    Returns:
        html.Div: A Dash component containing alert boxes or a message if there are no alerts.
    """
    alerts = fetch_alerts_from_redis()

    if not alerts:
        return html.Div("No current alerts")

    alert_boxes = [create_alert_box(alert, i) for i, alert in enumerate(alerts)]
    return html.Div(alert_boxes)


@app.callback(
    Output({"type": "alert-collapse", "index": MATCH}, "is_open"),
    Input({"type": "alert-toggle", "index": MATCH}, "n_clicks"),
    State({"type": "alert-collapse", "index": MATCH}, "is_open"),
)
def toggle_alert_collapse(n, is_open):
    """
    Toggles the collapse state of an alert box.

    Args:
        n (int): Number of clicks on the alert toggle button.
        is_open (bool): Current open state of the alert collapse.

    Returns:
        bool: New open state of the alert collapse.
    """
    if n:
        ctx = callback_context.triggered[0]
        prop_id = ctx["prop_id"]

        start_index = prop_id.find('"index":"') + 9
        end_index = prop_id.find('"', start_index)
        alert_id = prop_id[start_index:end_index]

        if is_open:
            expanded_alerts.discard(alert_id)
        else:
            expanded_alerts.add(alert_id)
        return not is_open
    return is_open


@app.callback(
    Output("system-explanation", "children"),
    Input("interval-component", "n_intervals"),
)
def update_system_explanation(n_intervals):
    return html.Div(
        [
            html.H5("About the System"),
            html.P(
                "This dashboard monitors the Particle Filter (PF) system, which estimates the state of various connectors including taps, pipes, and valves."
            ),
            html.P(
                "The system uses probability thresholds to generate alerts for potential issues, helping to prevent failures and maintain optimal performance."
            ),
        ]
    )


@app.callback(
    Output("dynamic-insights", "children"),
    Input("interval-component", "n_intervals"),
)
def update_dynamic_insights(n_intervals):
    """
    Updates the system explanation section.

    Args:
        n_intervals (int): Number of intervals (unused, but required for the callback).

    Returns:
        html.Div: A Dash component containing the system explanation.
    """
    global connector_probs, current_alerts

    total_connectors = len(connector_probs)
    high_risk_connectors = sum(1 for prob in connector_probs.values() if prob >= 0.2)
    low_risk_connectors = sum(
        1 for prob in connector_probs.values() if 0.1 <= prob < 0.2
    )

    return html.Div(
        [
            html.H5("System Insights"),
            html.Ul(
                [
                    html.Li(f"Total connectors monitored: {total_connectors}"),
                    html.Li(f"High-risk connectors: {high_risk_connectors}"),
                    html.Li(f"Low-risk connectors: {low_risk_connectors}"),
                    html.Li(f"Active alerts: {len(current_alerts)}"),
                ]
            ),
            html.P(
                f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                className="text-muted small",
            ),
        ]
    )


@app.callback(
    Output("system-alerts-content", "children"),
    Input("interval-component", "n_intervals"),
)
def update_system_alerts(n_intervals):
    """
    Updates the system alerts section.

    Args:
        n_intervals (int): Number of intervals (unused, but required for the callback).

    Returns:
        html.Div: A Dash component containing alert boxes or a message if there are no alerts.
    """
    alerts = fetch_alerts_from_redis()

    if not alerts:
        return html.Div("No current alerts")

    alert_boxes = [create_alert_box(alert, i) for i, alert in enumerate(alerts)]
    return html.Div(alert_boxes)


@app.callback(
    Output({"type": "recommendation-collapse", "index": MATCH}, "is_open"),
    Input({"type": "recommendation-toggle", "index": MATCH}, "n_clicks"),
    State({"type": "recommendation-collapse", "index": MATCH}, "is_open"),
)
def toggle_recommendation_collapse(n, is_open):
    """
    Toggles the collapse state of a recommendation box.

    Args:
        n (int): Number of clicks on the recommendation toggle button.
        is_open (bool): Current open state of the recommendation collapse.

    Returns:
        bool: New open state of the recommendation collapse.
    """
    if n:
        ctx = callback_context.triggered[0]
        prop_id = ctx["prop_id"]

        start_index = prop_id.find('"index":"') + 9
        end_index = prop_id.find('"', start_index)
        alert_id = prop_id[start_index:end_index]

        if is_open:
            expanded_recommendations.discard(alert_id)
        else:
            expanded_recommendations.add(alert_id)
        return not is_open
    return is_open


def redis_listener():
    """
    Listens for messages from Redis and updates the global connector probabilities and alerts.
    This function runs in a separate thread and continuously processes incoming Redis messages.
    """
    global connector_probs, current_alerts
    HIGH_PROB_THRESHOLD = 0.2
    LOW_PROB_THRESHOLD = 0.1

    print("Redis listener started. Waiting for messages...")
    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])

                connector_probs = {k: float(v) for k, v in data.items()}

                active_connectors = set()

                for connector, prob in connector_probs.items():
                    existing_alert = next(
                        (
                            alert
                            for alert in current_alerts
                            if alert["connector"] == connector
                        ),
                        None,
                    )
                    
                    if message['channel'] == b'recommendations':
                        pass

                    if prob >= HIGH_PROB_THRESHOLD:
                        alert_type = "high"
                        alert_content = {
                            "title": f"High Alert! {connector}",
                            "content": f"High breakage probability detected for connector: {connector}",
                            "recommendation": f"Urgent inspection and maintenance required for: {connector_to_inspect}",
                            "type": alert_type,
                            "model": "Probability Threshold Model",
                            "connector": connector,
                        }
                        active_connectors.add(connector)
                    elif LOW_PROB_THRESHOLD <= prob < HIGH_PROB_THRESHOLD:
                        alert_type = "low"
                        alert_content = {
                            "title": f"Low Alert! {connector}",
                            "content": f"Elevated breakage probability detected for connector: {connector}",
                            "recommendation": f"Schedule inspection for: {connector_to_inspect}",
                            "type": alert_type,
                            "model": "Probability Threshold Model",
                            "connector": connector,
                        }
                        active_connectors.add(connector)
                    else:
                        continue

                    if existing_alert:
                        if existing_alert["type"] != alert_type:
                            existing_alert.update(alert_content)
                            existing_alert["updated_at"] = time.time()
                            print(f"Updated alert for {connector}: {existing_alert}")
                    else:
                        new_alert = {
                            **alert_content,
                            "created_at": time.time(),
                            "updated_at": time.time(),
                        }
                        current_alerts.append(new_alert)
                        print(f"Added new alert: {new_alert}")

                current_alerts = [
                    alert
                    for alert in current_alerts
                    if alert["connector"] in active_connectors
                ]

                print(f"Current alerts after update: {len(current_alerts)}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON from Redis message: {message['data']}")
            except Exception as e:
                print(f"Error processing Redis message: {e}")
                print(f"Message data: {data}")


def fetch_alerts_from_redis() -> List[Dict[str, Any]]:
    """
    Fetches the current alerts from the global alerts list.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing alert information.
    """
    global current_alerts
    print(f"Fetching alerts. Current alerts: {len(current_alerts)}")
    return current_alerts


def check_redis_connection():
    """
    Periodically checks the Redis connection and prints the status.
    This function runs in a separate thread and checks the connection every 60 seconds.
    """
    while True:
        try:
            if r.ping():
                print("Redis connection is active")
            else:
                print("Redis connection failed")
        except redis.ConnectionError:
            print("Redis connection error")
        time.sleep(60)  # Check every 60 seconds


if __name__ == "__main__":
    # Start the Redis listener in a separate thread
    Thread(target=redis_listener, daemon=True).start()

    # Start the Redis connection checker in a separate thread
    Thread(target=check_redis_connection, daemon=True).start()

    # Run the Dash app
    app.run_server(debug=True, use_reloader=False, port=8051)
