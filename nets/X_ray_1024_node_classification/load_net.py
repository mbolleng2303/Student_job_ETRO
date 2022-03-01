"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.X_ray_1024_node_classification.graphsage_net import GraphSageNet


def GraphSage(net_params):

    return GraphSageNet(net_params)


def gnn_model(MODEL_NAME, net_params):

    models = {
        'GraphSage': GraphSage,
    }

    return models[MODEL_NAME](net_params)