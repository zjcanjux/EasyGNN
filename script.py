from asyncio.log import logger
import imp
from easygnn.dataset.academic_graph  import AcademicDataset
from easygnn.dataset import build_dataset
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


dataset = AcademicDataset("acm4HeCo",raw_dir="")



# result = build_dataset("acm4HeCo","node_classification",logger="")
# print(result.get_labels())


import graphviz

dot = graphviz.Digraph(comment="the round table")
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')
print(dot.source)

dot.render('graph_visualization/round-table.gv').replace('\\', '/')