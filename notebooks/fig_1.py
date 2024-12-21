import cairosvg
from graphviz import Digraph
from pyprojroot.here import here

wd = here()

# %%

# Create a graph object
dot = Digraph(comment='Overall Workflow')

# Add title with spacing
dot.attr(label="ClassFactory Workflow", labelloc="t", fontsize="24", fontname="Helvetica-Bold")

# Add an invisible spacer node
dot.node('Spacer', '', shape='point', width='0', height='0', style='invisible')

# Global Node Attributes
dot.attr('node', shape='box', style='rounded, filled', fillcolor='lightblue', fontname='Helvetica', fontsize='12', fixedsize='false')

# Helper function to define validator nodes (different color)


def add_validator_node(graph, node_id, label):
    graph.node(node_id, label, shape='egg',  style='filled', fillcolor='lightyellow')


# High-Level Workflow Nodes
dot.node('A', 'Load Lesson \nReadings', width='1.5', height='0.6')
dot.node('B', 'Extract Lesson \nObjectives', width='1.5', height='0.6')
dot.node('C', 'Select Module', width='1.5', height='0.6')
dot.node('D1', 'BeamerBot', width='1.5', fontname="Helvetica-Bold", penwidth='2')
dot.node('D2', 'ConceptWeb', width='1.5', fontname="Helvetica-Bold", penwidth='2')
dot.node('D3', 'QuizMaker', width='1.5',  fontname="Helvetica-Bold", penwidth='2')

# High-Level Workflow Edges
dot.edge('Spacer', 'A', style='invis')
dot.edges([('A', 'B'), ('B', 'C'), ('C', 'D1'), ('C', 'D2'), ('C', 'D3')])

# BeamerBot Sub-Flow
dot.node('D1.1', 'Load Prior Lesson', width='1.5', height='0.8')
dot.node('D1.2', 'Generate Slides', width='1.5', height='0.8')
add_validator_node(dot, 'D1.2.1', 'Validate \nLLM Response')
dot.node('D1.3', 'Render LaTeX Output', width='1.8', height='0.8')
dot.node('D1.4', 'Save Slides', width='2.0', height='0.8')
dot.edges([('D1', 'D1.1'), ('D1.1', 'D1.2'), ('D1.2', 'D1.2.1'), ('D1.2.1', 'D1.3'), ('D1.3', 'D1.4')])

# ConceptWeb Sub-Flow
dot.node('D2.1', 'Summarize Readings', width='1.8', height='0.8')
dot.node('D2.2', 'Extract Relationships', width='1.8', height='0.8')
add_validator_node(dot, 'D2.2.1', 'Validate \nLLM Response')
dot.node('D2.3', 'Entity Resolution', width='1.8', height='0.8')
dot.node('D2.4', 'Build Graph', width='1.8', height='0.8')
dot.node('D2.5', 'Detect Communities', width='1.8', height='0.8')
dot.node('D2.6', 'Visualize Graph', width='1.8', height='0.8')
dot.edges([('D2', 'D2.1'), ('D2.1', 'D2.2'), ('D2.2', 'D2.2.1'), ('D2.2.1', 'D2.3'),
           ('D2.3', 'D2.4'), ('D2.4', 'D2.5'), ('D2.5', 'D2.6')])

# QuizMaker Sub-Flow
dot.node('D3.1', 'Generate Questions', width='1.8', height='0.8')
add_validator_node(dot, 'D3.1.1', 'Validate \nLLM Response')
dot.node('D3.2', 'Remove Duplicates', width='1.8', height='0.8')
dot.node('D3.3', 'Export Quiz', width='1.8', height='0.8')
dot.node('D3.4', 'Host Quiz', width='1.8', height='0.8')
dot.node('D3.5', 'Analyze Results', width='1.8', height='0.8')
dot.edges([('D3', 'D3.1'), ('D3.1', 'D3.1.1'), ('D3.1.1', 'D3.2'), ('D3.2', 'D3.3'),
           ('D3.3', 'D3.4'), ('D3.4', 'D3.5')])

# Render the diagram in svg because other versions don't render correctly
dot.render('classfactory_workflow', format='svg', view=False, directory=wd/"reports/figures")
svg_path = wd/"reports/figures/classfactory_workflow.svg"

# convert to png for export
png_path = svg_path.with_suffix(".png")
cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
