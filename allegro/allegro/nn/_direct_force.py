from typing import List, Optional

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, Dropout

from e3nn.o3 import Irreps

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_runstats.scatter import scatter

import os
import torch.distributed as dist
import torch.multiprocessing as mp

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
        # bn = BatchNorm1d(dimensions[0])
        # force_predictor.append(bn)
        # dropout = Dropout(p=0.1)
        # force_predictor.append(dropout)
        for i in range(len(dimensions) - 1):
            linear = Linear(dimensions[i], dimensions[i + 1])
            if linear.bias is not None:
                linear.bias.data.fill_(0)
            force_predictor.append(linear)
            # bn = BatchNorm1d(dimensions[i + 1])
            # force_predictor.append(bn)
            force_predictor.append(ShiftedSoftplus())
        force_predictor.append(Linear(dimensions[-1], 1))
        self.force_predictor = Sequential(*force_predictor)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_emb = data["features_for_direct_force"]
        unit_vec = data["edge_vectors"] / data["edge_lengths"].view(-1, 1)
        edge_center = data["edge_index"][0]
        # if deploy for LAMMPS # TODO
        # edge_neighbor = data["edge_neighbors"]
        # edge_cell_shift = data["edge_cell_shift_lammps"].long()
        # atom_count = data["atom_count"][0]
        # else deploy for PyTorch # TODO
        edge_neighbor = data["edge_index"][1]
        edge_cell_shift = data["edge_cell_shift"].long()
        atom_count = data["atomic_energy"].shape[0]

        edge_lengths = data["edge_lengths"]
        edge_magnitude = self.force_predictor(edge_emb)
        edge_emb_sorted = torch.zeros((0, edge_emb.shape[-1]), dtype=edge_emb.dtype, device=edge_emb.device)
        unit_vec_sorted = torch.zeros((0, 3), dtype=unit_vec.dtype, device=unit_vec.device)
        edge_center_sorted = torch.zeros((0), dtype=edge_center.dtype, device=edge_center.device)
        edge_neighbor_sorted = torch.zeros((0), dtype=edge_neighbor.dtype, device=edge_neighbor.device)
        edge_lengths_sorted = torch.zeros((0), dtype=edge_neighbor.dtype, device=edge_neighbor.device)
        edge_magnitude_sorted = torch.zeros((0), dtype=edge_magnitude.dtype, device=edge_magnitude.device)
        edge_start = torch.zeros((7), dtype=torch.long, device=edge_center.device)
        edge_count = torch.zeros((7), dtype=torch.long, device=edge_center.device)
        for i in range(7):
            if i == 0: # local
                mask = (edge_cell_shift[:,0] == 0) & (edge_cell_shift[:,1] == 0) & (edge_cell_shift[:,2] == 0)
                sorted_index = torch.argsort(edge_center[mask] * atom_count + edge_neighbor[mask])
            elif i == 1: # ghost 后
                mask = (edge_cell_shift[:,0] == -1) & (edge_cell_shift[:,1] == 0) & (edge_cell_shift[:,2] == 0)
                sorted_index = torch.argsort(edge_center[mask] * atom_count + edge_neighbor[mask])
            elif i == 2: # ghost 左
                mask = (edge_cell_shift[:,1] == -1) & (edge_cell_shift[:,2] == 0)
                sorted_index = torch.argsort((edge_cell_shift[:,0][mask] + 2) * atom_count * atom_count + (edge_center[mask] * atom_count + edge_neighbor[mask]))
            elif i == 3: # ghost 下
                mask = (edge_cell_shift[:,2] == -1)
                sorted_index = torch.argsort((edge_cell_shift[:,0][mask] + 6) * (edge_cell_shift[:,1][mask] + 2) * atom_count * atom_count + (edge_center[mask] * atom_count + edge_neighbor[mask]))
            elif i == 4: # ghost 前
                mask = (edge_cell_shift[:,0] == 1) & (edge_cell_shift[:,1] == 0) & (edge_cell_shift[:,2] == 0)
                sorted_index = torch.argsort(edge_neighbor[mask] * atom_count + edge_center[mask])
            elif i == 5: # ghost 右
                mask = (edge_cell_shift[:,1] == 1) & (edge_cell_shift[:,2] == 0)
                sorted_index = torch.argsort((-edge_cell_shift[:,0][mask] + 2) * atom_count * atom_count + (edge_neighbor[mask] * atom_count + edge_center[mask]))
            else: # ghost 上
                mask = (edge_cell_shift[:,2] == 1)
                sorted_index = torch.argsort((-edge_cell_shift[:,0][mask] + 6) * (-edge_cell_shift[:,1][mask] + 2) * atom_count * atom_count + (edge_neighbor[mask] * atom_count + edge_center[mask]))
            edge_emb_sorted = torch.cat((edge_emb_sorted, torch.index_select(edge_emb[mask], 0, sorted_index)))
            unit_vec_sorted = torch.cat((unit_vec_sorted, torch.index_select(unit_vec[mask], 0, sorted_index)))
            edge_center_sorted = torch.cat((edge_center_sorted, torch.index_select(edge_center[mask], 0, sorted_index)))
            edge_neighbor_sorted = torch.cat((edge_neighbor_sorted, torch.index_select(edge_neighbor[mask], 0, sorted_index)))
            edge_lengths_sorted = torch.cat((edge_lengths_sorted, torch.index_select(edge_lengths[mask], 0, sorted_index)))
            edge_magnitude_sorted = torch.cat((edge_magnitude_sorted, torch.index_select(edge_magnitude[mask], 0, sorted_index)))
            edge_count[i] = torch.sum(mask)
            if i > 0:
                edge_start[i] = edge_start[i - 1] + edge_count[i - 1]
        sorted_index = torch.argsort(edge_neighbor_sorted[0 : int(edge_start[1])] * atom_count + edge_center_sorted[0 : int(edge_start[1])])
        edge_magnitude_sorted[0 : int(edge_start[1])] = (edge_magnitude_sorted[0 : int(edge_start[1])] + torch.index_select(edge_magnitude_sorted[0 : int(edge_start[1])], 0, sorted_index)) / 2
        # if deploy for LAMMPS # TODO
        data["forces"] = edge_magnitude_sorted # edge forces
        data["edge_centers"] = edge_center_sorted
        data["edge_unit_vec"] = unit_vec_sorted
        data["edge_start"] = edge_start
        data["edge_count"] = edge_count
        # else deploy for PyTorch # TODO
        edge_magnitude_sorted[int(edge_start[1]) : int(edge_start[4])] = (edge_magnitude_sorted[int(edge_start[1]) : int(edge_start[4])] + edge_magnitude_sorted[int(edge_start[4]) : ]) / 2
        edge_magnitude_sorted[int(edge_start[4]) : ] = edge_magnitude_sorted[int(edge_start[1]) : int(edge_start[4])]
        data["forces"] = scatter(edge_magnitude_sorted * unit_vec_sorted, edge_center_sorted, dim=0, reduce='sum')
        return data