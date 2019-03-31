#https://stackoverflow.com/questions/49794023/use-keras-model-with-flatten-layer-inside-opencv-3/49817506#49817506
import tensorflow as tf
from tensorflow.core import framework


def find_all_nodes(graph_def, **kwargs):
    for node in graph_def.node:
        for key, value in kwargs.items():
            if getattr(node, key) != value:
                break
        else:
            yield node
    raise StopIteration


def find_node(graph_def, **kwargs):
    try:
        return next(find_all_nodes(graph_def, **kwargs))
    except StopIteration:
        raise ValueError(
            'no node with attributes: {}'.format(
                ', '.join("'{}': {}".format(k, v) for k, v in kwargs.items())))


def walk_node_ancestors(graph_def, node_def, exclude=set()):
    openlist = list(node_def.input)
    closelist = set()
    while openlist:
        name = openlist.pop()
        if name not in exclude:
            node = find_node(graph_def, name=name)
            openlist += list(node.input)
            closelist.add(name)
    return closelist


def remove_nodes_by_name(graph_def, node_names):
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].name in node_names:
            del graph_def.node[i]


def make_shape_node_const(node_def, tensor_values):
    node_def.op = 'Const'
    node_def.ClearField('input')
    node_def.attr.clear()
    node_def.attr['dtype'].type = framework.types_pb2.DT_INT32
    tensor = node_def.attr['value'].tensor
    tensor.dtype = framework.types_pb2.DT_INT32
    tensor.tensor_shape.dim.add()
    tensor.tensor_shape.dim[0].size = len(tensor_values)
    for value in tensor_values:
        tensor.tensor_content += value.to_bytes(4, 'little')
    output_shape = node_def.attr['_output_shapes']
    output_shape.list.shape.add()
    output_shape.list.shape[0].dim.add()
    output_shape.list.shape[0].dim[0].size = len(tensor_values)


def make_cv2_compatible(graph_def):
    # A reshape node needs a shape node as its second input to know how it
    # should reshape its input tensor.
    # When exporting a model using Keras, this shape node is computed
    # dynamically using `Shape`, `StridedSlice` and `Pack` operators.
    # Unfortunately those operators are not supported yet by the OpenCV API.
    # The goal here is to remove all those unsupported nodes and hard-code the
    # shape layer as a const tensor instead.
    for reshape_node in find_all_nodes(graph_def, op='Reshape'):

        # Get a reference to the shape node
        shape_node = find_node(graph_def, name=reshape_node.input[1])

        # Find and remove all unsupported nodes
        garbage_nodes = walk_node_ancestors(graph_def, shape_node,
                                            exclude=[reshape_node.input[0]])
        remove_nodes_by_name(graph_def, garbage_nodes)

        # Infer the shape tensor from the reshape output tensor shape
        if not '_output_shapes' in reshape_node.attr:
            raise AttributeError(
                'cannot infer the shape node value from the reshape node. '
                'Please set the `add_shapes` argument to `True` when calling '
                'the `Session.graph.as_graph_def` method.')
        output_shape = reshape_node.attr['_output_shapes'].list.shape[0]
        output_shape = [dim.size for dim in output_shape.dim]

        # Hard-code the inferred shape in the shape node
        make_shape_node_const(shape_node, output_shape[1:])
