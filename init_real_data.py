# First Party Library
from data_loader import (
    attr_graph_dynamic_spmat_DBLP,
    attr_graph_dynamic_spmat_NIPS,
    attr_graph_dynamic_spmat_twitter,
    spmat2sptensor,
)

TOTAL_TIME = 10
# input_graph = attr_graph_dynamic_spmat_twitter(T=TOTAL_TIME)
# input_graph = attr_graph_dynamic_spmat_NIPS(T=TOTAL_TIME)
input_graph = attr_graph_dynamic_spmat_DBLP(T=TOTAL_TIME)


class LoadDataset:
    adj = []
    feature = []

    def __init__(self, adj, feature):
        self.adj = adj
        self.feature = feature


def init_real_data() -> LoadDataset:
    adj = input_graph.Gmat_list
    feature = input_graph.Amat_list

    for t in range(input_graph.T):
        adj[t] = adj[t]
        _ = spmat2sptensor(adj[t])

        adj[t] = _
        feature_ = input_graph.Amat_list[t]

        _ = spmat2sptensor(feature_)
        feature[t] = _

    return LoadDataset(
        adj=adj,
        feature=feature,
    )
