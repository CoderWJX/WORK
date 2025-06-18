from typing import List, Optional

import torch
from torch.nn import Linear, Sequential

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_runstats.scatter import scatter

class DirectForce(GraphModuleMixin, torch.nn.Module):
    """
    direct force v3
    """
    def __init__(
            self,
            dimensions: List[int],
            field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: Optional[str] = None,
            irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        force_predictor = []
        for i in range(len(dimensions) - 1):
            force_predictor.append(Linear(dimensions[i], dimensions[i + 1]))
            force_predictor.append(ShiftedSoftplus())
        force_predictor.append(Linear(dimensions[-1], 1))
        self.force_predictor = Sequential(*force_predictor)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # 1. predict force for every edge
        edge_emb = data["features_for_direct_force"]
        unit_vec = data["edge_vectors"] / data["edge_lengths"].view(-1, 1)
        edge_center = data["edge_index"][0]
        edge_neighbor = data["edge_index"][1]
        edge_lengths = data["edge_lengths"]
        edge_magnitude = self.force_predictor(edge_emb)
        atom_bias = scatter(edge_magnitude.T, edge_center, dim=0, reduce='mean')
        force_magnitude_bias = torch.zeros(edge_magnitude.shape[0], device=unit_vec.device)
        # torch.scatter(force_magnitude_bias, 0, edge_center, atom_bias)
        for i in range(0, atom_bias.shape[0]):
            edge_magnitude[edge_center == i] = edge_magnitude[edge_center == i] - atom_bias[i]
        # edge_magnitude = edge_magnitude - force_magnitude_bias
        # del atom_bias, force_magnitude_bias
        # 2. Add edges for ghost atoms (LAMMPS simulation)
        max_edge_center = edge_center[-1]
        ghost_atom_mask = edge_neighbor > max_edge_center
        unit_vec = torch.concat((unit_vec, -1.0 * unit_vec[ghost_atom_mask]), dim=0)
        edge_center_to_concat = edge_neighbor[ghost_atom_mask]
        edge_neighbor_to_concat = edge_center[ghost_atom_mask]
        edge_center = torch.concat((edge_center, edge_center_to_concat), dim=0)
        edge_neighbor = torch.concat((edge_neighbor, edge_neighbor_to_concat), dim=0)
        edge_lengths = torch.concat((edge_lengths, edge_lengths[ghost_atom_mask]), dim=0)
        edge_magnitude = torch.concat((edge_magnitude, edge_magnitude[ghost_atom_mask]), dim=0)
        # torch.set_printoptions(profile="full")
        # print(torch.concat((edge_center.view(-1,1), edge_neighbor.view(-1,1)), dim=1))
        # print(edge_lengths.view(-1, 1))
        # 3. search edge pairs
        with torch.no_grad():
            edge_index_add_length = edge_center + edge_neighbor + \
                                    (10000000000 * edge_lengths).long() + \
                                    (10000000000 * unit_vec.abs().sum(dim=1)).long()
            if edge_index_add_length.shape[0] % 2 != 0:
                # find edge pairs for odd egde numbers case
                sorted_edge_length, index = torch.sort(edge_index_add_length)
                tmp1 = torch.concat((torch.zeros((1, 3), device=unit_vec.device),
                                     unit_vec[index][0:index.shape[0]-1,:]), dim=0)
                tmp2 = torch.concat((unit_vec[index][1:index.shape[0],:],
                                     torch.zeros((1, 3), device=unit_vec.device)), dim=0)
                flag1 = torch.any(torch.ne((unit_vec[index] + tmp1), 0), dim=1)
                flag2 = torch.any(torch.ne((unit_vec[index] + tmp2), 0), dim=1)
                mask = (~flag1) | (~flag2)   # or ^?
                mask_2 = (~flag1) & (~flag2)
                if torch.nonzero(mask_2).numel() != 0:
                    if index[mask].shape[0] % 2 != 0:
                        print(torch.nonzero(mask_2))
                        mask[torch.nonzero(mask_2)] = False
                edge_pairs = index[mask].view(-1, 2)

            else:
                edge_pairs = torch.argsort(edge_index_add_length).view(-1, 2)

        # 4. enforce F_st = F_ts
        edge_magnitude[edge_pairs[:,0]] = (torch.index_select(edge_magnitude, 0, edge_pairs[:,0])
                                           + torch.index_select(edge_magnitude, 0, edge_pairs[:,1])) / 2
        edge_magnitude[edge_pairs[:,1]] = torch.index_select(edge_magnitude, 0, edge_pairs[:,0])
        # 5. sum forces
        edge_force = edge_magnitude * unit_vec
        data["edge_force"] = edge_force
        data["forces"] = scatter(edge_force, edge_center, dim=0, reduce='sum')
        forces_pad = torch.zeros((data["pos"].shape[0] - data["forces"].shape[0], 3), device=data["forces"].device)
        data["forces"] = torch.concat((data["forces"], forces_pad))
        return data
