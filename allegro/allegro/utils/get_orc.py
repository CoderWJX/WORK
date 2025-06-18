from torch_geometric.utils import to_networkx
from allegro.utils.OllivierRicci import OllivierRicci

import numpy as np


def _get_single_graph_statistics(G):
    orc = OllivierRicci(G, alpha=0)
    orc.compute_ricci_curvature()

    all_curvatures = []
    for i, j in orc.G.edges:
        all_curvatures.append(orc.G[i][j]['ricciCurvature']['rc_curvature'])
    all_curvatures = np.array(all_curvatures)

    return all_curvatures.mean(), all_curvatures.std()


# data_statistics = {}
# for key in datasets['node_cls']:
#     print(f'[INFO] Calculating curvatures for {key}')
#     dataset = datasets['node_cls'][key]
#     G = to_networkx(dataset.data)
#     mean, std = _get_single_graph_statistics(G)
#
#     data_statistics[key] = {
#         'mean' : mean,
#         'std' : std
#     }

# for key in datasets['graph_cls']:
def get_orc(datasets):
    # print(f'[INFO] Calculating curvatures for {key}')
    # dataset = datasets['graph_cls'][key]
    avg_curvatures = []
    print("datasets lengrh : " + len(datasets))
    for i in range(len(datasets)):
        G = to_networkx(datasets[i])
        mean, _ = _get_single_graph_statistics(G)
        avg_curvatures.append(mean)
    avg_curvatures = np.array(avg_curvatures)
    # data_statistics[key] = {
    #     'mean' : avg_curvatures.mean(),
    #     'std' : avg_curvatures.std()
    # }
    return avg_curvatures.mean(), avg_curvatures.std()


# # Display curvatures
# df = pd.DataFrame(data=data_statistics)
# print(df)