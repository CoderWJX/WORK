import numpy as np
import torch
import matplotlib.pyplot as plt

data_path = "/mnt/yujie.zeng/project_2023/lammps_output.zip"
lammps_out = torch.load(data_path)

def draw_atoms(pos, index, real_atom_num, edge_neig):
    cutoff = 6
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # draw real atoms
    ax.scatter(pos[0:real_atom_num-1, 0], pos[0:real_atom_num-1, 1], pos[0:real_atom_num-1, 2], marker='*', color='red', label="real atoms")
    # draw rmax area, not sure how to calculate s
    ax.scatter(pos[0:real_atom_num-1, 0], pos[0:real_atom_num-1, 1], pos[0:real_atom_num-1, 2], s=7600*np.pi, marker='o', color='red', alpha=0.05)
    # find outliers
    pos_neih = np.empty((0, 3))
    pos_outlier = np.empty((0, 3))
    ghost_num = pos.shape[0] - real_atom_num
    for i in range(real_atom_num, pos.shape[0]):
        if i in edge_neig:
            pos_neih = np.append(pos_neih, pos[i])
        else:
            pos_outlier = np.append(pos_outlier, pos[i])
    pos_neih= np.reshape(pos_neih, (-1, 3))
    pos_outlier= np.reshape(pos_outlier, (-1, 3))

    # draw neighbor ghost atoms
    ax.scatter(pos_neih[:, 0], pos_neih[:, 1], pos_neih[:, 2], marker='^', color = 'green', label="neighbor ghost atoms" )
    # draw outlier ghost atoms
    ax.scatter(pos_outlier[:, 0], pos_outlier[:, 1], pos_outlier[:, 2], marker='*', label="outlier ghost atoms")
    print("ghost_neighbor number: %d (%f)"% ( pos_neih.shape[0], pos_neih.shape[0]/ghost_num))
    print("ghost_outlier number: %d (%f)"% ( pos_outlier.shape[0], pos_outlier.shape[0]/ghost_num))


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()
    return


pos = lammps_out['pos'].cpu().numpy()
edge_neig = lammps_out['edge_index'][1].cpu().numpy()
index = np.arange(0, pos.shape[0])
real_atom_num = 28
ghost_atom_num = pos.shape[0] - real_atom_num
draw_atoms(pos, index, real_atom_num, edge_neig)
# print(lammps_out)


