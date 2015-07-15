from pystruct.models import GraphCRF, EdgeFeatureGraphCRF

def make_edges(n_nodes):
        return np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]

X_graph = np.array([(x, make_edges(len(x))) for x in X])
X_graph_train, X_graph_test = X_graph[folds == 1], X_graph[folds != 1]


graph_model = GraphCRF(inference_method="max-product", directed=True)
ssvm = FrankWolfeSSVM(model=graph_model, C=.1, max_iter=11)
ssvm.fit(X_graph_train, y_train)
print("score with GraphCRF %f" % ssvm.score(X_graph_test, y_test))


X_edge_features = np.array([(x, make_edges(len(x)), np.ones(len(x)) - 1)[:, np.newaxis] for x in X])
X_edge_features_train, X_edge_features_test = X_edge_features[folds == 1], X_edge_features[folds != 1]

edge_feature_model = EdgeFeatureGraphCRF(inference_method="max-product")
ssvm = FrankWolfeSSVM(model=edge_feature_model, C=.1, max_iter=11)
ssvm.fit(X_edge_features_train, y_train)
print("score with GraphCRF %f" % ssvm.score(X_edge_features_test, y_test))
