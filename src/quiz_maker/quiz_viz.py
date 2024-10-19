from pathlib import Path

import pandas as pd
# plot setup
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html


def generate_dashboard(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Generate an interactive dashboard using Dash to display summary statistics and plots.

    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
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
            dcc.Graph(figure=item['figure'])
        ], style={
            'width': '23%',          # Adjust width for 3-4 plots per row
            'display': 'inline-block',
            'verticalAlign': 'top',
            'margin': '1%',
            'boxSizing': 'border-box'
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
            'justifyContent': 'space-around'
        })
    ])

    # Run the Dash app
    app.run_server(debug=False)
    print("Access server at\nhttp://127.0.0.1:8050")


def generate_html_report(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, quiz_df: pd.DataFrame = None) -> None:
    """
    Generate a static HTML report containing summary statistics and plots.

    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
        summary (pd.DataFrame): DataFrame containing summary statistics.
        output_dir (Path): Directory where the report will be saved.
        quiz_df (pd.DataFrame, optional): DataFrame containing the quiz questions and options.
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
    Create a Plotly figure for a specific question.

    Args:
        df (pd.DataFrame): DataFrame containing quiz responses.
        question_text (str): The question text to plot.
        quiz_df (pd.DataFrame, optional): DataFrame containing the quiz questions and options.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Filter data for the question
    question_df = df[df['question'] == question_text]
    # Get the correct answer
    correct_answer = question_df['correct_answer'].iloc[0]

    # Determine all possible options
    if quiz_df is not None:
        # Get the options for the question from quiz_df
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
                        option_labels.append(col.strip(')'))
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
        # Get unique options from user answers and correct answer
        options = set(question_df['user_answer'].unique()).union(set([correct_answer]))
        possible_options = sorted(options)

    # Create a DataFrame with all possible options
    options_df = pd.DataFrame({'user_answer': possible_options})

    # Count the number of each answer
    answer_counts = question_df.groupby('user_answer').size().reset_index(name='counts')

    # Merge with options_df to include zero counts
    answer_counts = pd.merge(options_df, answer_counts, on='user_answer', how='left').fillna(0)

    # Ensure counts are integers
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
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Update legend labels
    fig.for_each_trace(lambda t: t.update(name='Correct' if t.name == 'True' else 'Incorrect'))

    return fig
