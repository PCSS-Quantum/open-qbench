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