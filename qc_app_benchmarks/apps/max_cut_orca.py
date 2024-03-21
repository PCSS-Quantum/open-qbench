def max_cut_thetas_3_edges(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 3 nodes and 2 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1), "input_state2": (0, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 3 nodes and 2 edges", {"input_state1": (1, 1, 1), "input_state2": (0, 1, 1)}
    return [-1.7229, -1.1057]

def max_cut_thetas_4_edges(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 4 nodes and 4 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1, 1), "input_state2": (0, 1, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 4 nodes and 4 edges", {"input_state1": (1, 1, 1, 1), "input_state2": (0, 1, 1, 1)}
    return [1.9212, 1.8164, 1.3147]

def max_cut_thetas_6_edges(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 6 nodes and 10 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 10 edges", {"input_state1": (1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1)}
    return [-3.0492, -0.1812, -0.7512, -2.1761,  0.2920]

def max_cut_thetas_7_edges(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 7 nodes and 16 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 10 edges", {"input_state1": (1, 1, 1, 1, 1, 1, 1),
                                                   "input_state2": (0, 1, 1, 1, 1, 1, 1)}
    return [-1.5334,  0.0372,  0.8819, -1.9504,  0.6715,  2.6831]

def max_cut_5_edges_new_input(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 5 nodes and 6 edges"
    if return_input_state:
        return {"input_state1": (1, 0, 1, 0, 1), "input_state2": (1, 0, 1, 0, 1)}
    if return_graph and return_input_state:
        return "Graph with 5 nodes and 6 edges", {"input_state1": (1, 0, 1, 0, 1), "input_state2": (1, 0, 1, 0, 1)}
    return [-0.7304,  1.5838,  0.0172,  1.2462]

def max_cut_5_edges_new_input_double_loop(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 5 nodes and 8 edges"
    if return_input_state:
        return {"input_state1": (1, 0, 1, 0, 1), "input_state2": (1, 0, 1, 0, 1)}
    if return_graph and return_input_state:
        return "Graph with 5 nodes and 8 edges", {"input_state1": (1, 0, 1, 0, 1), "input_state2": (1, 0, 1, 0, 1)}
    return [-2.5717, -0.8906,  2.1858,  0.5305,  3.0327,  2.1154, -0.0739,  0.4309]

def max_cut_6_edges_new_input(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 6 nodes and 13 edges"
    if return_input_state:
        return {"input_state1": (1, 0, 1, 0, 1, 0), "input_state2": (1, 0, 1, 0, 1, 0)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 13 edges", {"input_state1": (1, 0, 1, 0, 1, 0), "input_state2": (1, 0, 1, 0, 1, 0)}
    return [0.8479, -0.0095,  0.2154, -1.3921,  0.0614]

def max_cut_6_edges_new_input_double_loop(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 6 nodes and 11 edges"
    if return_input_state:
        return {"input_state1": (1, 0, 1, 0, 1, 0), "input_state2": (1, 0, 1, 0, 1, 0)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 11 edges", {"input_state1": (1, 0, 1, 0, 1, 0), "input_state2": (1, 0, 1, 0, 1, 0)}
    return [-0.3523, -0.4798,  2.8224,  0.7053, -3.0474,  0.3007, -1.5556, -1.9770,
         -0.9119, -0.4838]

def max_cut_thetas_6_edges_double_loop(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 6 nodes and 11 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 10 edges", {"input_state1": (1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1)}
    return [ 1.1061, -2.8851,  3.0852,  0.4741, -1.4476,  3.0600,  1.0738,  1.6214,
         -0.8227,  2.1824]

def max_cut_thetas_7_edges_double_loop(return_graph=False, return_input_state=False):
    if return_graph:
        return "Graph with 7 nodes and 15 edges"
    if return_input_state:
        return {"input_state1": (1, 1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1, 1)}
    if return_graph and return_input_state:
        return "Graph with 6 nodes and 10 edges", {"input_state1": (1, 1, 1, 1, 1, 1), "input_state2": (0, 1, 1, 1, 1, 1)}
    return [-1.1602,  2.9214,  1.6905,  2.0678, -0.5040,  1.8149,  2.8871,  2.4254,
         -0.7878,  1.5720,  1.4504, -1.2389]
