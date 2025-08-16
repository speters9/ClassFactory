"""
quiz_viz.py
-----------

This module provides visualization utilities for quiz results, including:
- An interactive dashboard (using Dash and Plotly) to display summary statistics and per-question analysis.
- Generation of static HTML reports with embedded plots and summary tables for quiz assessments.

Key Functions:
- `generate_dashboard`: Launches a Dash web app for interactive quiz result exploration.
- `generate_html_report`: Creates a static HTML report with summary statistics and per-question plots.
- `create_question_figure`: Generates Plotly figures for individual quiz questions.

Dependencies: pandas, plotly, dash, jinja2 (for HTML reports).
"""

from pathlib import Path

import pandas as pd
# plot setup
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html


def generate_dashboard(df: pd.DataFrame, summary: pd.DataFrame, test_mode: bool = False) -> None:
    """
    Launch an interactive dashboard using Dash to display quiz summary statistics and per-question plots.


    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
        test_mode (bool): If True, does not launch the server (for testing purposes).
        summary (pd.DataFrame): DataFrame containing summary statistics.
    """
    app = Dash(__name__)
    # Create a list of plotly figures for each question
    figures = []
    for question in df['question'].unique():
        fig = create_question_figure(df, question)
        figures.append({'question': question, 'figure': fig})

    # Create a list of dcc.Graph components
    graphs = []
    for item in figures:
        graph = html.Div([
            html.H3(item['question'], style={'textAlign': 'center'}),
            dcc.Graph(figure=item['figure'], style={'height': '350px'})
        ], style={
            'width': '23%',          # Adjust width for 3-4 plots per row
            # Removed 'display: inline-block' to avoid stacking issues
            'verticalAlign': 'top',
            'margin': '1%',
            'boxSizing': 'border-box',
            'height': '420px',  # Fixed height for each graph container
            'overflow': 'hidden',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'flex-start'
        })
        graphs.append(graph)

    # Layout of the dashboard
    app.layout = html.Div([
        html.H1('Quiz Assessment Dashboard', style={'textAlign': 'center'}),
        html.H2('Summary Statistics'),
        dash_table.DataTable(
            data=summary.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in summary.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
        ),
        html.H2('Question Analysis'),
        html.Div(children=graphs, style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'space-around',
            'alignItems': 'flex-start',
            'width': '100%'
        })
    ])

    # Run the Dash app
    if not test_mode:
        app.run(debug=False)
        print("Access server at\nhttp://127.0.0.1:8050")


def generate_html_report(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, quiz_df: pd.DataFrame = None) -> None:
    """
    Generate a static HTML report with summary statistics and per-question plots for quiz results.

    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
        summary (pd.DataFrame): DataFrame containing summary statistics.
        output_dir (Path): Directory where the report will be saved.
        quiz_df (pd.DataFrame, optional): DataFrame containing the quiz questions and options (for option text in plots).
    """
    from jinja2 import Environment, FileSystemLoader

    # Prepare plots
    plots = []
    for question in df['question'].unique():
        fig = create_question_figure(df, question, quiz_df)
        # Convert the figure to HTML div string
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        plots.append({'question': question, 'plot_html': plot_html})

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))
    template = env.from_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quiz Assessment Report</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .plot-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
            }
            .plot-item {
                width: 23%;
                margin: 1%;
                box-sizing: border-box;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
            h2 {
                margin-top: 40px;
            }
            @media screen and (max-width: 768px) {
                .plot-item {
                    width: 45%;
                }
            }
            @media screen and (max-width: 480px) {
                .plot-item {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Quiz Assessment Report</h1>
        <h2>Summary Statistics</h2>
        {{ summary_table }}
        <h2>Question Analysis</h2>
        <div class="plot-container">
        {% for item in plots %}
            <div class="plot-item">
                <h3 style="text-align: center;">{{ item.question }}</h3>
                {{ item.plot_html | safe }}
            </div>
        {% endfor %}
        </div>
    </body>
    </html>
    ''')
    # Convert summary DataFrame to HTML table with styling
    summary_table = summary.to_html(index=False, classes='summary-table')

    # Render the template
    html_content = template.render(summary_table=summary_table, plots=plots)

    # Save the report to an HTML file
    report_path = output_dir / 'quiz_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Report saved to '{report_path}'")


def create_question_figure(df: pd.DataFrame, question_text: str, quiz_df: pd.DataFrame = None) -> go.Figure:
    """
    Create a Plotly bar chart for a specific quiz question, showing answer distribution and correctness.

    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
        question_text (str): The question text to plot.
        quiz_df (pd.DataFrame, optional): DataFrame containing the quiz questions and options (for answer text mapping).

    Returns:
        go.Figure: Plotly figure object visualizing answer distribution for the question.
    """
    # Standardize answer keys in user_answer and correct_answer to be 'A', 'B', 'C', 'D'
    def standardize_key(val):
        if isinstance(val, str):
            v = val.strip()
            if v.endswith(')') and len(v) == 2 and v[0] in 'ABCD':
                return v[0]
            if v in 'ABCD':
                return v
        return val

    question_df = df[df['question'] == question_text].copy()
    question_df['user_answer'] = question_df['user_answer'].apply(standardize_key)
    correct_answer = standardize_key(question_df['correct_answer'].iloc[0])

    # Determine all possible options
    if quiz_df is not None:
        quiz_question = quiz_df[quiz_df['question'] == question_text]
        if not quiz_question.empty:
            option_columns = ['A)', 'B)', 'C)', 'D)']
            options_list = []
            option_labels = []
            for col in option_columns:
                if col in quiz_question.columns:
                    option_text = str(quiz_question.iloc[0][col])
                    if pd.notna(option_text) and option_text.strip() != '':
                        options_list.append(option_text.strip())
                        option_labels.append(col[0])  # 'A)' -> 'A'
        else:
            options_list = None
            option_labels = None
    else:
        options_list = None
        option_labels = None

    # Determine possible options
    if option_labels is not None:
        possible_options = option_labels
    else:
        question_df['user_answer'] = question_df['user_answer'].fillna('No Answer')
        possible_options = sorted(set(question_df['user_answer'].unique()).union(set([correct_answer])))

    options_df = pd.DataFrame({'user_answer': possible_options})
    answer_counts = question_df.groupby('user_answer').size().reset_index(name='counts')
    answer_counts = pd.merge(options_df, answer_counts, on='user_answer', how='left').fillna(0)
    answer_counts['counts'] = answer_counts['counts'].astype(int)

    # Determine if each answer is correct
    answer_counts['is_correct'] = answer_counts['user_answer'] == correct_answer

    # Map user_answer to option_text if available
    if options_list is not None and option_labels is not None:
        option_map = dict(zip(option_labels, options_list))
        answer_counts['option_text'] = answer_counts['user_answer'].map(option_map)
    else:
        answer_counts['option_text'] = answer_counts['user_answer']

    # Create the bar chart
    fig = px.bar(
        answer_counts,
        x='user_answer',
        y='counts',
        color='is_correct',
        color_discrete_map={True: 'green', False: 'red'},
        labels={'user_answer': 'Answer Option', 'counts': 'Number of Responses'},
        category_orders={'user_answer': possible_options}
    )
    # Update hover data to include option_text
    fig.update_traces(
        hovertemplate='Option: %{x}<br>Option Text: %{customdata[0]}<br>Responses: %{y}',
        customdata=answer_counts[['option_text']].values
    )
    fig.update_layout(
        xaxis_title='Answer Options',
        yaxis_title='Number of Responses',
        title_x=0.5,
        legend_title_text='Correct Answer',
        margin=dict(l=20, r=20, t=40, b=20),
        height=400  # Fix the height to prevent dashboard from growing
    )
    # Update legend labels
    fig.for_each_trace(lambda t: t.update(name='Correct' if t.name == 'True' else 'Incorrect'))

    return fig
