from .Tree import Tree
from .Node import Node


class BinaryTree(Tree):
    def __init__(self, name='undefined'):
        super().__init__(name=name)

    def addChild(self, parent, child, **kwargs):
        raise Exception("For BinaryTree, please use addLeft or addRight.")

    def addLeft(self, parent, child, weight=1, color='black', width=0.003, properties=None):
        properties = dict(properties or {})
        properties['side'] = 'left'
        if self.getLeft(parent) is not None:
            raise Exception("Left child already exists for this parent.")
        return super().addChild(parent, child, weight=weight, color=color, width=width, properties=properties)

    def addRight(self, parent, child, weight=1, color='black', width=0.003, properties=None):
        properties = dict(properties or {})
        properties['side'] = 'right'
        if self.getRight(parent) is not None:
            raise Exception("Right child already exists for this parent.")
        return super().addChild(parent, child, weight=weight, color=color, width=width, properties=properties)

    def getLeft(self, parent):
        for edge in parent.edges_out:
            if edge.properties.get('side') == 'left':
                return edge.end
        return None

    def getRight(self, parent):
        for edge in parent.edges_out:
            if edge.properties.get('side') == 'right':
                return edge.end
        return None

    def getChildren(self, parent):
        children = []
        left = self.getLeft(parent)
        right = self.getRight(parent)
        if left is not None:
            children.append(left)
        if right is not None:
            children.append(right)
        return children

    def __str__(self):
        root_key = self.root.key if self.root else 'None'
        return f"BinaryTree (root: {root_key})\n" + super().__str__()