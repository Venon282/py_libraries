import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle
import random
import numpy as np
from collections import deque

from .Node import Node
from .Edge import Edge

class Graph:
    def __init__(self, name='undefined',
                 is_directed=True, show_labels_max_nodes=150):
        self.name = name
        self.nodes = {} 
        self.edges = set()
        self.is_directed = is_directed
        
        self.show_labels_max_nodes=show_labels_max_nodes
    
    # region Construct Graph
    @staticmethod
    def constructByKeyPairs(pairs):
        # def _getNode(key, memory, graph):
        #     if key in memory:
        #         node = memory[key]
        #     else:
        #         node = Node(key)
        #         memory[key] = node
        #         graph.nodes.add(node)
        #     return node
        
        def _getNode(key, graph):
            if key in graph.nodes:
                node = graph.nodes[key]
            else:
                node = Node(key)
                graph.nodes[key] = node
            return node
        
        #element_created = {}
        graph = Graph()
        for key1, key2 in pairs:
            key1, key2 = str(key1), str(key2)
            node1 = _getNode(key1, graph)
            node2 = _getNode(key2, graph)
            edge = Edge(node1, node2)
            graph.edges.add(edge)
            node1.addEdge(edge)
            node2.addEdge(edge)
            
        return graph
    # endregion
    
    # region Updates
    
    def addNode(self, node):
        """
        Adds a node to the graph.
        Parameters:
            node (Node): The node instance to add.
        """
        if node.key not in self.nodes:
            #self.nodes.add(node)
            self.nodes[node.key] = node
            
    def removeNode(self, node):
        """
        Removes a node from the graph, along with any edges connected to it.
        Parameters:
            node (Node): The node instance to remove.
        """
        #self.nodes.discard(node)
        
        if self.nodes.pop(node.key, None) is None:
            return
    
        # Remove all edges that are connected to this node.
        self.edges = self.edges - node.edges # {edge for edge in self.edges if edge not in node.edges}
        
        for edge in node.edges:
            # Determine which node is the "neighbor"
            neighbor = edge.end if edge.start is node else edge.start
            
            # Guard against self-loops
            if neighbor is not node:
                neighbor.edges.discard(edge)
                
        node.edges.clear()

        
    def addEdge(self, edge):
        """
        Adds an edge to the graph and also registers the edge with the corresponding nodes.
        Parameters:
            edge (Edge): The edge instance to add.
        """
        # Ensure both nodes are part of the graph.
        self.addNode(edge.start)
        self.addNode(edge.end)
        
        self.edges.add(edge)
        
        # Ensure both nodes have this edge in memory
        edge.start.addEdge(edge)
        edge.end.addEdge(edge)
        
    def removeEdge(self, edge):
        """
        Removes an edge from the graph.
        Parameters:
            edge (Edge): The edge instance to remove.
        """
        self.edges.remove(edge)
        edge.start.removeEdge(edge)
        edge.end.removeEdge(edge)
        
    def discardEdge(self, edge):
        """
        Discard an edge from the graph.
        Parameters:
            edge (Edge): The edge instance to discard.
        """
        self.edges.discard(edge)
        edge.start.discardEdge(edge)
        edge.end.discardEdge(edge)
    # end region
    
    # region Is
    def isLinkedList(self):
        num_nodes = len(self.nodes)
        
        # empty case
        if num_nodes == 0:
            return False
        
        # one node case
        if num_nodes == 1:
            return len(next(iter(self.nodes.values())).edges) == 0

        head_node = None
        tail_node = None

        for node in self.nodes.values():
            out_degree = node.degree_out
            in_degree = node.degree_in

            if out_degree == 1 and in_degree == 0:
                if head_node: return False # Two heads
                head_node = node
            elif out_degree == 0 and in_degree == 1:
                if tail_node: return False # Two tails
                tail_node = node
            elif out_degree == 1 and in_degree == 1:
                continue # Middle node, looks good
            else:
                return False # Branching or isolated node

        if not head_node or not tail_node:
            return False # It's a cycle or disconnected

        # Final check: Is it all one connected component?
        # Walk from head to tail to ensure no disconnected circles exist
        visited_count = 0
        curr = head_node
        while curr:
            visited_count += 1
            # Find the next node
            next_edge = next((e for e in curr.edges if e.start is curr), None)
            curr = next_edge.end if next_edge else None

        return visited_count == num_nodes
    
    def isTree(self):
        n_nodes = len(self.nodes)
        
        if n_nodes == 0:
            return False
        if n_nodes == 1:
            return next(iter(self.nodes.values())).degree_in == 0
    
        root = None
        for node in self.nodes.values():
            if node.degree_in > 1: # graph have multiple parents
                return False
            elif node.degree_in == 0:
                if root: return False # case a second root
                root = node

        queue = deque([root])
        visited = {root.key}

        while queue:
            node = queue.popleft()
            
            for edge in node.edges_out:
                neighbor = edge.end
                
                if neighbor.key not in visited:
                    visited.add(neighbor.key)
                    queue.append(neighbor)
                else:
                    return False
            
        return len(visited) == n_nodes
            
    # end region
    
    # region Get        
    def getEdgesPosition(self):
        start_xs = self.edgesStartX
        start_ys = self.edgesStartY
        end_xs = self.edgesEndX
        end_ys = self.edgesEndY
        start_radius = self.edgesStartRadius
        end_radius = self.edgesEndRadius
        
        dxs = end_xs - start_xs
        dys = end_ys - start_ys
        
        distance = np.hypot(dxs, dys)
        distance_mask = distance == 0
        distance[distance_mask] = 1
        
        x_normalisation = dxs / distance
        y_normalisation = dys / distance
        
        start_x = start_xs + x_normalisation * start_radius
        start_y = start_ys + y_normalisation * start_radius

        end_x = end_xs - x_normalisation * end_radius
        end_y = end_ys - y_normalisation * end_radius
            
        return (start_x, start_y), (end_x, end_y)
    
    def getNodeByKey(self, key):
        """
        Retrieves a node by its name.
        Parameters:
            name (str): The name identifier of the node.
        Returns:
            Node or None: The node with the matching name, or None if not found.
        """
        # for node in self.nodes:
        #     if node.key == key:
        #         return node
        return self.nodes.get(key, None)

    # end region
    
    # region Visualisation
    def drawNodes(self, fig, ax):
        def updateLabelVisibility(event=None):
            """ 
            Allow to display the labels only if it have a limited quantity of nodes display because it's an eavy operation
            """
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            # Check the visible nodes
            visible_nodes = {node.key for node, txt in texts if x0 <= node.x <= x1 and y0 <= node.y <= y1}
            
            # Does some labels can be display ?
            show = len(visible_nodes) <= self.show_labels_max_nodes

            # Define if the label must be display or not
            for node, txt in texts:
                txt.set_visible(show and node.key in visible_nodes)

            fig.canvas.draw_idle()
        # collection.set_picker(True) #! allow to know which node or dge will be click
        patches = [Circle((node.x, node.y), node.radius) for node in self.nodes.values()]
        node_colors = self.nodesColor
        edge_colors = self.nodesEdgeColor
        edge_widths = self.nodesEdgeWidth
        texts = [(node, ax.text(
                    node.x,
                    node.y,
                    str(node.key),
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=3,
                    fontweight='bold',
                    visible=False  # start hidden
                ))for node in self.nodes.values()]
        

        collection = PatchCollection(
            patches,
            facecolor=node_colors,
            edgecolor=edge_colors,
            linewidths=edge_widths,
        )

        ax.add_collection(collection)
        
        ax.callbacks.connect('xlim_changed', updateLabelVisibility)
        ax.callbacks.connect('ylim_changed', updateLabelVisibility)
        updateLabelVisibility()
        
    def drawEdges(self, ax):
        (edge_start_x, edge_start_y), (edge_end_x, edge_end_y) = self.getEdgesPosition()
        
        if self.is_directed: # arrow if direct graph

            dx = edge_end_x - edge_start_x
            dy = edge_end_y - edge_start_y
  
            ax.quiver(
                edge_start_x,
                edge_start_y,
                dx,
                dy,
                angles='xy',
                scale_units='xy',
                scale=1,
                width=next(iter(self.edges)).width,
                color=self.edgesColor,
            )
        else:
            starts = np.stack((edge_start_x, edge_start_y), axis=1)  
            ends   = np.stack((edge_end_x, edge_end_y), axis=1)  

            segments = np.stack((starts, ends), axis=1)

            lc = LineCollection(segments, colors=self.edgesColor, linewidths=self.edgesWidth)
            ax.add_collection(lc)
        
    def draw(self, fig_size=(10, 8)):
        def defineLimits(ax):
            xs = self.nodesX
            ys = self.nodesY
            rs = self.nodesRadius

            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            
            x_margin = (max_x-min_x) *0.01
            y_margin = (max_y-min_y) *0.01
            ax.set_xlim(min_x-x_margin, max_x+x_margin)
            ax.set_ylim(min_y-y_margin, max_y+y_margin)
            
        
            
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Handle ax
        ax.invert_yaxis() # (0, 0) become upper start
        ax.xaxis.tick_top()
        ax.axis('off')
        
        self.drawNodes(fig, ax)
        self.drawEdges(ax)
        
        defineLimits(ax)   

        plt.tight_layout()
        plt.show()
    # endregion
        
    # region Properties
    @property
    def nodesX(self):
        return np.array([n.x for n in self.nodes.values()])
    
    @property
    def nodesY(self):
        return np.array([n.y for n in self.nodes.values()])
    
    @property
    def nodesRadius(self):
        return np.array([n.radius for n in self.nodes.values()])
    
    @property
    def nodesColor(self):
        return np.array([n.color for n in self.nodes.values()])
    
    @property
    def nodesEdgeColor(self):
        return np.array([n.edge_color for n in self.nodes.values()])
    
    @property
    def nodesEdgeWidth(self):
        return np.array([n.edge_width for n in self.nodes.values()])
    
    @property
    def edgesStartX(self):
        return np.array([e.start.x for e in self.edges])
    
    @property
    def edgesStartY(self):
        return np.array([e.start.y for e in self.edges])
    
    @property
    def edgesStartRadius(self):
        return np.array([e.start.radius for e in self.edges])
    
    @property
    def edgesEndX(self):
        return np.array([e.end.x for e in self.edges])
    
    @property
    def edgesEndY(self):
        return np.array([e.end.y for e in self.edges])
    
    @property
    def edgesEndRadius(self):
        return np.array([e.end.radius for e in self.edges])

    @property
    def edgesDX(self):
        return self.edgesEndX - self.edgesStartX
    
    @property
    def edgesDY(self):
        return self.edgesEndY - self.edgesStartY
    
    @property
    def edgesCoordinate(self):
        return np.array([e.coordinate for e in self.edges])
    
    @property
    def edgesColor(self):
        return np.array([e.color for e in self.edges])
    
    @property
    def edgesWidth(self):
        return np.array([e.width for e in self.edges])


    # endregion
 
    # region Cast
    def toDict(self):
        """
        Returns a dictionary representation of the graph.

        Returns:
            dict: A dictionary with lists of nodes and edges.
        """
        return {
            "nodes": [node.toDict() for node in self.nodes.values()],
            "edges": [edge.toDict() for edge in self.edges]
        }
    # end region
    
    # region overwriting
    def __str__(self):
        """
        Returns a string representation of the graph.

        Returns:
            str: A summary of the graph's nodes and edges.
        """
        nodes_str = ', '.join(self.nodes.keys())
        edges_str = ', '.join([
            f"{edge.start.key}->{edge.end.key}" if self.is_directed else f"{edge.start.name}-{edge.end.name}"
            for edge in self.edges
        ])
        return f"--- Graph {self.name} ---\nNodes: {nodes_str}\nEdges: {edges_str}"
    # end region
