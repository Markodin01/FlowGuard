"""
PF GUI Dash Application

This script creates a modern, interactive dashboard for the PF (Pipe Flow) system
using Dash and Plotly. It visualizes the system state, allows for user control
of various parameters, and displays system alerts and repair team statuses.

The application uses Redis for real-time updates and includes features such as
start/stop controls, system reset, and interactive graphs.
"""

import sys
import os
import time
import json
import queue
import threading
from typing import Dict, List, Any

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, ALL, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import redis
import jsonpickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import Source, Sink, TapPipe, ValvePipe, LowAlarm, HighAlarm
from core.system import System
from core.tp_parser import parse



# Initialize Redis connection
redis_client = redis.StrictRedis(host="localhost", port=6379)
pubsub = redis_client.pubsub()
pubsub.subscribe("valve_action")

# Shared queue for thread-safe operations
message_queue = queue.Queue()

# Global variables
FNAME = None
if len(sys.argv) == 3:
    FNAME = sys.argv[2]

# Initialize Dash app with a modern theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Custom CSS for additional styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f8f9fa;
            }
            .card {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .card-header {
                background-color: #007bff;
                color: white;
            }
            .btn-primary {
                background-color: #007bff;
                border-color: #007bff;
            }
            .btn-primary:hover {
                background-color: #0056b3;
                border-color: #0056b3;
            }
            .navbar {
                box-shadow: 0 2px 4px rgba(0,0,0,.1);
            }
            .alert-card {
                border: none;
                overflow: hidden;
            }
            .alert-card .card-header {
                border-bottom: none;
            }
            .alert-card .btn-link {
                text-decoration: none;
            }
            .alert-card .card-body {
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
def system_to_1d_array(system: System) -> List[int]:
    """
    Convert the system state to a 1D array.

    Args:
        system (System): The current state of the system.

    Returns:
        List[int]: 1D array representing the system state [valve states, alarms].
    """
    valve_states = [
        int(valve.desired_flow / valve.max_flow * 5)
        for valve in system.get_all_connectors().values()
        if isinstance(valve, (TapPipe, ValvePipe))
    ]
    
    alarm_states = []
    for tank in system.get_all_tanks().values():
        if not isinstance(tank, (Source, Sink)):
            if isinstance(tank.triggered_alarm, HighAlarm):
                alarm_states.append(2)
            elif isinstance(tank.triggered_alarm, LowAlarm):
                alarm_states.append(1)
            else:
                alarm_states.append(0)
    
    return valve_states + alarm_states



def write_file(s: System) -> None:
    """
    Write the system state to a file.

    Args:
        s (System): The current state of the system.
    """
    a = []
    if os.path.isfile(FNAME):
        with open(FNAME, "r") as f:
            a = jsonpickle.decode(f.read())
            a.append(s)
    with open(FNAME, "w") as f:
        f.write(jsonpickle.encode(a))


def get_node_positions(s: System, fixed_node_pos: Dict[str, tuple]) -> Dict[Any, tuple]:
    """
    Calculate node positions for the system graph.

    Args:
        s (System): The current state of the system.
        fixed_node_pos (Dict[str, tuple]): Dictionary of fixed node positions.

    Returns:
        Dict[Any, tuple]: Dictionary of calculated node positions.
    """
    g = nx.DiGraph()
    for k, c in s.get_all_connectors().items():
        g.add_edge(c.source, c.sink)

    pos = {}
    fixed = []

    for p, l in fixed_node_pos.items():
        pos[s.get_tank(p)] = l
        fixed.append(s.get_tank(p))

    for k, t in s.get_all_tanks():
        if isinstance(t, Source) and k not in pos:
            pos[t] = (0, 0.5)
            fixed.append(t)
        elif isinstance(t, Sink) and k not in pos:
            pos[t] = (1, 0.5)
            fixed.append(t)

    return nx.spring_layout(g, pos=pos, fixed=fixed, seed=2)

def make_sliders(system: System) -> List[dbc.Row]:
    """
    Create slider components for system controls.

    Args:
        system (System): The current state of the system.

    Returns:
        List[dbc.Row]: List of Dash Bootstrap rows containing sliders.
    """
    return [
        dbc.Row(
            [
                dbc.Col(html.Label(k, className="font-weight-bold"), width=3),
                dbc.Col(
                    dcc.Slider(
                        min=c.min_flow,
                        max=c.max_flow,
                        value=c.desired_flow,
                        step=(1 if isinstance(c, TapPipe) else c.max_flow - c.min_flow),
                        id={"type": "slider", "index": f"{k}"},
                        marks={
                            i: str(i)
                            for i in range(
                                int(c.min_flow),
                                int(c.max_flow) + 1,
                                max(1, int((c.max_flow - c.min_flow) / 5)),
                            )
                        },
                    ),
                    width=9,
                ),
            ],
            className="mb-3",
        )
        for k, c in sorted(system.get_all_connectors().items())
        if isinstance(c, (TapPipe, ValvePipe))
    ]



# Load system configuration
system, node_pos, mspertick = parse(sys.argv[1])
tank_node_pos = get_node_positions(system, node_pos)
for name, tank in system.get_all_tanks().items():
    node_pos[name] = tank_node_pos[tank]

initial_sliders = make_sliders(system)
def make_graph(s: System, node_pos: Dict[str, tuple]) -> go.Figure:
    """
    Create a Plotly graph object representing the system.

    Args:
        s (System): The current state of the system.
        node_pos (Dict[str, tuple]): Dictionary of node positions.

    Returns:
        go.Figure: Plotly figure object representing the system graph.
    """

    if s is None or node_pos is None:
        # Return an empty figure if data is not available
        return go.Figure()

    # Create a mapping from tank names to positions
    np = {s.get_tank(n): pos for n, pos in node_pos.items()}

    graph_layout = np
    source_pos = None
    sink_pos = None
    tankx = []
    tanky = []
    tanktext = []
    tank_colors = []
    alarm_shapes = []

    # Create a mapping from tanks to their alarms
    tank_to_alarms = {tank: [] for tank in s.get_all_tanks().values()}
    for alarm in s.get_all_alarms().items():
        # TODO: remove this hack (alarm[1])
        if alarm[1].tank in tank_to_alarms:
            tank_to_alarms[alarm[1].tank].append(alarm)

    # Iterate over tanks to prepare graph data
    for k, t in s.get_all_tanks().items():
        if isinstance(t, Source):
            source_pos = graph_layout[t]
        elif isinstance(t, Sink):
            sink_pos = graph_layout[t]
        else:
            tankx.append(graph_layout[t][0])
            tanky.append(graph_layout[t][1])
            tanktext.append(
                f"{k}<br>{t.contains}/{t.capacity}"
                if t.visible
                else f"{k}<br>?/{t.capacity}"
            )
            tank_colors.append("#ffffff")  # Default color if no alarms are triggered
            for alarm in tank_to_alarms[t]:
                if alarm[1].triggered:
                    shape = create_alarm_shape(
                        graph_layout[t][0], graph_layout[t][1], alarm
                    )
                    if shape:
                        alarm_shapes.append(shape)

    tanks_fig = go.Scatter(
        x=tankx,
        y=tanky,
        mode="markers",
        marker=dict(
            size=50,
            color=tank_colors,
            line=dict(color="Black", width=1),
            symbol="square",
        ),
    )
    source_fig = go.Scatter(
        x=[source_pos[0]],
        y=[source_pos[1]],
        mode="markers",
        marker=dict(size=50, color="#00ff00", symbol="circle"),
    )
    sink_fig = go.Scatter(
        x=[sink_pos[0]],
        y=[sink_pos[1]],
        mode="markers",
        marker=dict(size=50, color="#3F71FF", symbol="circle"),
    )

    arrows_fig = []
    edge_figs = []
    edge_annotations = []
    for k, c in s.get_all_connectors().items():
        x = (graph_layout[c.source][0], graph_layout[c.sink][0])
        y = (graph_layout[c.source][1], graph_layout[c.sink][1])
        # text=f"{k}<br>{c.min_flow}/{c.max_flow}"
        text = f"{k}<br>{c.desired_flow}"

        if hasattr(c, '_broken') and c._broken:
            line_color = "red"
            line_width = 2
            dash_style = "dash"
            text += "<br>[BROKEN]"
        else:
            line_color = "#AAAAAA"
            line_width = 0.5
            dash_style = None

        edge_figs.append(
            go.Scatter(
                x=x, 
                y=y, 
                mode="lines", 
                line=dict(
                    width=line_width, 
                    color=line_color,
                    dash=dash_style
                )
            )
        )
        
        arrows_fig.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=line_color,
                    symbol="arrow",
                    size=20,
                    angleref="previous",
                    standoff=30,
                ),
            )
        )

        edge_annotations.append([(x[0] + x[1]) / 2, (y[0] + y[1]) / 2, f"{text}"])

    fig = go.Figure(data=edge_figs + arrows_fig + [tanks_fig, source_fig, sink_fig])

    for i in range(len(tankx)):
        fig.add_annotation(
            x=tankx[i], y=tanky[i], text=tanktext[i], showarrow=False, yshift=0
        )

    for e in edge_annotations:
        fig.add_annotation(x=e[0], y=e[1], text=e[2], showarrow=False, yshift=0)

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        shapes=alarm_shapes,
        yaxis_range=[source_pos[1] - 0.5, source_pos[1] + 0.5],
    )
    1
    return fig


def create_alarm_shape(x: float, y: float, alarm: tuple) -> Dict[str, Any]:
    """
    Create a shape to represent an alarm on the graph.

    Args:
        x (float): X-coordinate of the alarm.
        y (float): Y-coordinate of the alarm.
        alarm (tuple): Tuple containing alarm information.

    Returns:
        Dict[str, Any]: Dictionary representing the alarm shape.
    """
    # Determine the alarm type (high or low) and set the shape's color and position
    if isinstance(alarm[1], LowAlarm):
        color = "rgba(255, 0, 0, 0.7)"  # Semi-transparent red for low alarms
        # The rectangle covers the bottom half of the tank marker
        y0 = y - 0.04  # Adjust these values based on your scale
        y1 = y
    elif isinstance(alarm[1], HighAlarm):
        color = "rgba(255, 165, 0, 0.7)"  # Semi-transparent orange for high alarms
        # The rectangle covers the top half of the tank marker
        y0 = y + 0.04
        y1 = y  # Adjust these values based on your scale
    else:
        return None  # No shape for unrecognized alarm types

    return {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "x0": x - 0.02,  # Adjust based on the scale
        "y0": y0,
        "x1": x + 0.02,  # Adjust based on the scale
        "y1": y1,
        "fillcolor": color,
        "opacity": 0.7,
        "line": {
            "width": 0,
        },
    }




def make_broken_connectors_div(system: System) -> html.Div:
    """Create a component displaying broken connectors."""
    broken_connectors = [
        (k, c) for k, c in system.get_all_connectors().items() 
        if hasattr(c, '_broken') and c._broken
    ]
    
    if not broken_connectors:
        return html.Div("No broken connectors")
    
    return html.Div([
        dbc.Alert(
            [
                html.Strong(f"Connector: {k} "),
                html.Span(f"Flow rate: {c._current_flow}"),
                html.Div([
                    f"Repair time remaining: {max(0, c.repair_time)}",  # Ensure non-negative
                    html.Span(
                        " (Repair in progress)" if c.repairTeam is not None else "",
                        style={"color": "green", "font-weight": "bold"}
                    )
                ])
            ],
            color="danger",
            className="mb-2",
        )
        for k, c in broken_connectors
    ])

def make_alarms_div(system: System) -> List[dbc.Alert]:
    """
    Create alert components for system alarms.

    Args:
        system (System): The current state of the system.

    Returns:
        List[dbc.Alert]: List of Dash Bootstrap alert components.
    """
    tank_ids = {v: k for k, v in system.get_all_tanks().items()}
    return [
        dbc.Alert(
            [
                dbc.Row(
                    [
                        dbc.Col(f"Time: {a[1].trigger_time}", width=6),
                        dbc.Col(f"Tank: {tank_ids[a[1].tank]}", width=6),
                        dbc.Col(
                            f"{'Low' if isinstance(a[1], LowAlarm) else 'High'} Level: {a[1].level}",
                            width=4,
                        ),
                    ]
                )
            ],
            color="danger" if isinstance(a[1], LowAlarm) else "warning",
            className="mb-2",
        )
        for a in system.get_all_alarms().items()
        if a[1].triggered
    ]


def make_repair_div(system: System) -> html.Div:
    """Create repair team display and controls with improved styling"""
    loc_options = [{"label": "Select a location...", "value": "None"}]
    
    # Add connector options
    for k, c in sorted(system.get_all_connectors().items()):
        is_broken = hasattr(c, '_broken') and c._broken
        # Include all connectors in options but mark broken ones
        label = f"{k} [BROKEN]" if is_broken else k
        loc_options.append({"label": label, "value": k})
    
    # Get a mapping from connectors to their IDs
    connector_to_id_map = {c: k for k, c in system.get_all_connectors().items()}
    
    rows = []
    for i, (team_id, team) in enumerate(system.get_all_repair_teams().items()):
        # Determine current assigned location
        current_location = "None"
        if team.location is not None:
            # Only set the current location if the connector is actually broken
            # This ensures UI resets even if RepairTeam object wasn't properly reset
            for k, c in system.get_all_connectors().items():
                if c == team.location:
                    if hasattr(c, '_broken') and c._broken:
                        current_location = k
                    else:
                        # If connector is not broken, force "None"
                        current_location = "None"
                    break
        
        # Create status message
        if team.location is None or current_location == "None":
            status_message = "Status: Idle"
        else:
            # Check if location is actually broken
            is_broken = hasattr(team.location, '_broken') and team.location._broken
            repair_time = getattr(team.location, 'repair_time', 0)
            
            if is_broken and repair_time > 0:
                status_message = f"Status: Repairing {connector_to_id_map.get(team.location, 'Unknown')} ({repair_time} ticks left)"
            elif is_broken:
                status_message = f"Status: Repairing {connector_to_id_map.get(team.location, 'Unknown')}"
            else:
                # Location not broken - force idle status
                status_message = "Status: Idle"
        
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(f"Team {i + 1}", className="fw-bold"), 
                        width=2,
                        className="d-flex align-items-center"
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id={"type": "repair_dropdown", "index": str(i)},
                            options=loc_options,
                            value=current_location,
                            className="mb-0",
                            clearable=False,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            id={"type": "repair_status", "index": str(i)},
                            children=status_message,
                            className="text-muted",
                        ),
                        width=4,
                        className="d-flex align-items-center"
                    ),
                ],
                className="mb-3",
            )
        )
    
    return html.Div(rows, id="repair_div", style={"maxHeight": "200px", "overflowY": "auto"})


control_panel = dbc.Card(
    [
        dbc.CardHeader("System Controls"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Start",
                                id="start-stop-button",
                                color="primary",
                                className="mr-2",
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Button("Reset", id="reset-button", color="secondary"),
                            width=2,
                        ),
                        dbc.Col(
                            html.Div("Time: ", className="font-weight-bold"), width=3
                        ),
                        dbc.Col(html.Div("0", id="current_time"), width=1),
                        dbc.Col(
                            html.Div("Score: ", className="font-weight-bold"), width=3
                        ),
                        dbc.Col(html.Div("0", id="current_score"), width=1),
                    ],
                    className="mb-3",
                ),
            ]
        ),
    ],
    className="mb-4",
)

sliders_panel = dbc.Card(
    [
        dbc.CardHeader("Control Sliders"),
        dbc.CardBody(
            [
                html.Div(id="sliders_div"),
            ]
        ),
    ],
    className="mb-4",
)

alarms_panel = dbc.Card(
    [
        dbc.CardHeader("Alarms"),
        dbc.CardBody(
            [
                html.Div(
                    id="alarms_div",
                    style={"height": "200px", "overflowY": "auto"},
                )
            ]
        ),
    ]
)

broken_connectors_panel = dbc.Card(
    [
        dbc.CardHeader("Broken Connectors"),
        dbc.CardBody(
            html.Div(id="broken_connectors_div", style={"maxHeight": "200px", "overflowY": "auto"})
        ),
    ],
    className="mb-4",
)

# Invisible components
invisible_components = [
    dcc.Store(id="node_positions", data=jsonpickle.encode(node_pos)),
    dcc.Store(id="datastore", data=jsonpickle.encode(system)),
    dcc.Interval(id="sim_timer", interval=mspertick, n_intervals=0, disabled=True),
]

# Main layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="graph",
                            figure=make_graph(system, node_pos),
                            style={"height": "60vh"},
                            config={"staticPlot": True},
                        ),
                    ],
                    md=7,
                ),
                dbc.Col(
                    [
                        control_panel,
                        sliders_panel,
                    ],
                    md=5,
                ),
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    [
                        alarms_panel,
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Repair Teams"),
                                dbc.CardBody(
                                    make_repair_div(system)
                                )
                            ]
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col([broken_connectors_panel], md=12),
            ],
            className="mt-4",
        ),
        
        *invisible_components,
        dcc.Store(id="buffered-slider-values", data={}),
    ],
    fluid=True,
    className="mt-4",
)
app.layout.children.append(dcc.Interval(id="redis-publish-interval", interval=10000, n_intervals=0))


def listen_to_redis():
    """
    Listen for messages from Redis and add them to the message queue.
    This function runs in a separate thread.
    """
    while True:
        message = pubsub.get_message()
        if message and message["type"] == "message":
            data = json.loads(message["data"])
            message_queue.put(data)


# Start Redis listener in a separate thread
listener_thread = threading.Thread(target=listen_to_redis, daemon=True)
listener_thread.start()

@app.callback(
    Output("redis-publish-interval", "disabled"),
    Input("start-stop-button", "children")
)
def toggle_redis_publish(button_text):
    return button_text == "Start"

@app.callback(
    Output("redis-publish-interval", "n_intervals"),
    Input("redis-publish-interval", "n_intervals"),
    State("datastore", "data")
)
def publish_system_state(n_intervals, system_data):
    if n_intervals > 0:
        system = jsonpickle.decode(system_data)
        state_array = system_to_1d_array(system)
        redis_client.set("system_state", json.dumps(state_array))
        print(f"Published system state to Redis: {state_array}")
    return n_intervals

@app.callback(
    Output("sliders_div", "children", allow_duplicate=True),
    Input("datastore", "data"),
    prevent_initial_call=True
)
def update_sliders(system_data):
    """Generate sliders based on current system state"""
    system = jsonpickle.decode(system_data)
    return make_sliders(system)

@callback(
    Output("datastore", "data", allow_duplicate=True),
    Input({"type": "repair_dropdown", "index": ALL}, "value"),
    State({"type": "repair_dropdown", "index": ALL}, "id"),
    State("datastore", "data"),
    prevent_initial_call=True,
)
def handle_repair_updates(repair_values, repair_ids, system_data):
    """
    Update repair team assignments based on dropdown selections.
    """
    system = jsonpickle.decode(system_data)
    ctx = dash.callback_context
    
    # Only update if the user actually changed a dropdown
    if not ctx.triggered:
        return no_update
    
    changed = False
    for i, repair_id in enumerate(repair_ids):
        if i < len(repair_values):
            team_index = repair_id['index']
            repair_team = system.get_repair_team(str(int(team_index) + 1))
            new_location_id = repair_values[i]
            
            # Set the new location
            new_location = None
            if new_location_id != "None":
                new_location = system.get_connector(new_location_id)
            
            # Only change if needed
            current_location_id = "None"
            if repair_team.location is not None:
                for k, c in system.get_all_connectors().items():
                    if c == repair_team.location:
                        current_location_id = k
                        break
            
            if current_location_id != new_location_id:
                repair_team.dispatch(new_location)
                changed = True
    
    if changed:
        return jsonpickle.encode(system)
    else:
        return no_update

@callback(
    # All outputs from the three callbacks
    Output("start-stop-button", "children"),
    Output("sim_timer", "disabled"),
    Output("start-stop-button", "n_clicks"),
    Output("start-stop-button", "disabled"),
    Output("sim_timer", "n_intervals"),
    Output("sliders_div", "children", allow_duplicate=True),
    Output("datastore", "data"),
    Output("alarms_div", "children"),
    Output("broken_connectors_div", "children"),
    Output("graph", "figure"),
    Output("current_time", "children"),
    Output("current_score", "children"),
    
    # All inputs that can trigger actions
    Input("start-stop-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    Input("sim_timer", "n_intervals"),
    
    # All states needed
    State("datastore", "data"),
    State("node_positions", "data"),
    State({"type": "slider", "index": ALL}, "value"),
    State({"type": "slider", "index": ALL}, "id"),
    
    prevent_initial_call=True,
)
def handle_ui_actions(
    start_btn_clicks, reset_btn_clicks, timer_intervals,
    system_data, node_pos_data, slider_values, slider_ids
):
    """
    Unified callback to handle all major UI actions: start/stop, reset, and simulation ticks.
    Uses callback_context to determine which input triggered the callback.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
        
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Get current system state
    system = jsonpickle.decode(system_data)
    node_pos = jsonpickle.decode(node_pos_data)
    
    # Default values - will use no_update where appropriate
    button_text = no_update
    timer_disabled = no_update
    button_clicks = no_update
    button_disabled = no_update
    timer_intervals_value = no_update
    sliders = no_update
    system_state = no_update
    alarms_div = no_update
    broken_div = no_update
    graph = no_update
    time_display = no_update
    score_display = no_update
    
    # Handle different triggers
    if triggered_id == "reset-button":
        # Reset logic
        global FNAME
        FNAME = str(time.time())  # Start a new file on reset
        
        # Parse the original file to get a fresh system
        fresh_system, _, _ = parse(sys.argv[1])
        fresh_system.reset()
        
        # Reset all important state
        for connector in fresh_system.get_all_connectors().values():
            if hasattr(connector, 'desired_flow'):
                connector.desired_flow = connector.min_flow
            
            # Make sure no connectors are broken
            if hasattr(connector, '_broken'):
                connector._broken = False
                connector.repair_time = 0
                connector.repairTeam = None
        
        # Reset all repair teams
        for repair_team in fresh_system.get_all_repair_teams().values():
            repair_team.location = None
            repair_team.moving = False
        
        # Reset all tanks to empty
        for tank in fresh_system.get_all_tanks().values():
            if not isinstance(tank, (Source, Sink)):
                tank.contains = 0
                tank.overflowed = False
        
        # Reset all alarms
        for alarm in fresh_system.get_all_alarms().values():
            alarm.reset()
        
        # Reset time and score
        fresh_system.time = 0
        fresh_system.score = 0
        
        # Update all outputs for reset
        button_text = "Start"
        timer_disabled = True
        button_clicks = 0
        button_disabled = False
        timer_intervals_value = 0
        sliders = make_sliders(fresh_system)
        system_state = jsonpickle.encode(fresh_system)
        alarms_div = make_alarms_div(fresh_system)
        graph = make_graph(fresh_system, node_pos)
        time_display = "0"
        score_display = "0"
        
    elif triggered_id == "start-stop-button":
        # Just toggle the button state and timer
        button_text = "Stop" if start_btn_clicks % 2 == 1 else "Start"
        timer_disabled = start_btn_clicks % 2 == 0
        
    elif triggered_id == "sim_timer":

        for team_id, team in system.get_all_repair_teams().items():
            if team.location is not None and (not hasattr(team.location, '_broken') or not team.location._broken):
                print(f"Auto-updating repair team {team_id}: connector fixed or not broken")
                team.location = None
                team.moving = False

        # Get buffered slider values from the store
        buffered_values = dash.callback_context.states.get('buffered-slider-values.data', {})
        
        # Apply buffered slider values
        for connector_id, value in buffered_values.items():
            if connector_id in system.get_all_connectors():
                system.get_connector(connector_id).desired_flow = value
            # Simulation tick logic
        if timer_intervals == 0:
            system.reset()

        
        # Update connector flows based on slider values
        for i, slider_id in enumerate(slider_ids):
            if i < len(slider_values):  # Safety check
                connector_id = slider_id["index"]
                value = slider_values[i]
                system.get_connector(connector_id).desired_flow = value
        
        # Process Redis messages
        while not message_queue.empty():
            valve_action = message_queue.get_nowait()
            for slider_id, slider_value in zip(slider_ids, slider_values):
                connector_id = slider_id["index"]
                if not any(prefix in connector_id for prefix in ["PP", "RP"]):
                    system.get_connector(connector_id).desired_flow = (
                        slider_value if slider_value == 1 else 0
                    )
        
        # For debugging, track broken connectors
        broken_before = [k for k, c in system.get_all_connectors().items() 
                         if hasattr(c, '_broken') and c._broken]
        
        # Standard simulation tick with breaking enabled
        system.score += 1
        disable_sim = system.tick(timer_intervals, breaking=True)
        
        # Track which connectors are still broken after
        broken_after = [k for k, c in system.get_all_connectors().items() 
                        if hasattr(c, '_broken') and c._broken]
        
        # Log if there's a change in broken status
        if set(broken_before) != set(broken_after):
            print(f"Time {timer_intervals}, Broken connectors changed:")
            print(f"  Before: {broken_before}")
            print(f"  After: {broken_after}")
        
        # Update return values based on tick results
        system_state = jsonpickle.encode(system)
        graph = make_graph(system, node_pos)
        alarms_div = make_alarms_div(system)
        time_display = str(system.time)
        score_display = str(system.score)
        broken_div = make_broken_connectors_div(system)
        
        # Publish system state to Redis if needed
        state_array = system_to_1d_array(system)
        redis_client.set("system_state", json.dumps(state_array))
        
        # Handle simulation end
        if disable_sim:
            button_text = "Start"
            timer_disabled = True
            button_disabled = True
    
    # Return all outputs in the correct order
    return (
        button_text,
        timer_disabled,
        button_clicks,
        button_disabled,
        timer_intervals_value,
        sliders,
        system_state,
        alarms_div,
        broken_div,
        graph,
        time_display,
        score_display
    )


@app.callback(
    Output("buffered-slider-values", "data"),
    Input({"type": "slider", "index": ALL}, "value"),
    State({"type": "slider", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def buffer_slider_changes(slider_values, slider_ids):
    """
    Buffer slider changes until the next simulation tick.
    """
    buffered_values = {}
    for value, slider_id in zip(slider_values, slider_ids):
        connector_id = slider_id["index"]
        buffered_values[connector_id] = value
    
    return buffered_values


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)