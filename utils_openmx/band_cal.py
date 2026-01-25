'''
Descripttion: The script to calculat bands from the results of HamGNN
version: 1.0
Author: Yang Zhong
Date: 2022-12-20 14:08:52
LastEditors: Hao-Jen You
LastEditTime: 2026-01-25 20:30:00
'''

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.core.periodic_table import Element
import math
import os
from utils_openmx.utils import *
import argparse
import yaml
import torch
import time

def parse_kpath_file(filepath):
    """
    Revised parsing function:
    1. Filters out points with identical consecutive coordinates to prevent redundant labels.
    2. Handles discontinuous paths represented by empty lines (e.g., jumps like H|A).
    """
    k_path = []
    raw_labels = []
    nk_per_segment = 20
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find kpath file: {filepath}")

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        
        # Read k-point density
        try: 
            nk_per_segment = int(lines[1])
        except: 
            pass

        content_lines = lines[4:]
        i = 0
        while i < len(content_lines):
            line = content_lines[i]
            
            # Handle jump points (discontinuous paths indicated by empty lines)
            if not line:
                if i > 0 and i + 1 < len(content_lines):
                    prev_parts = content_lines[i-1].split()
                    next_parts = content_lines[i+1].split()
                    if len(prev_parts) >= 4 and len(next_parts) >= 4:
                        prev_coords = [float(x) for x in prev_parts[:3]]
                        next_coords = [float(x) for x in next_parts[:3]]
                        
                        # Discontinuity is only valid if coordinates are actually different
                        if not np.allclose(prev_coords, next_coords):
                            if len(k_path) > 0: 
                                # Pop the duplicate old point to prepare for combined label
                                k_path.pop()
                                raw_labels.pop()
                            
                            combined = f"{prev_parts[3].replace('GAMMA', 'Γ')}|{next_parts[3].replace('GAMMA', 'Γ')}"
                            k_path.append(next_coords)
                            raw_labels.append(rf"$\mathrm{{{combined}}}$")
                            i += 2 # Skip the next line as it has been processed
                            continue
                i += 1
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                coords = [float(x) for x in parts[:3]]
                lb = parts[3].replace('GAMMA', 'Γ')
                
                # IMPORTANT: If coordinates are identical to the previous point, skip adding the label
                if len(k_path) > 0 and np.allclose(coords, k_path[-1]):
                    i += 1
                    continue
                
                k_path.append(coords)
                raw_labels.append(rf"$\mathrm{{{lb}}}$")
            i += 1
            
    return k_path, raw_labels, nk_per_segment

def plot_band_structure(k_dist, eigen, k_node, label, node_index, save_path):
    """
    Plot band structure using a professional publication-quality template.
    
    Parameters:
    k_dist (np.ndarray): Array of cumulative k-path distances.
    eigen (np.ndarray): Array of eigenvalues/energies (shape: [nbands, nk]).
    k_node (list): X-coordinates of high-symmetry points.
    label (list): Labels for high-symmetry points (handles Γ and H|A formatting).
    node_index (list): Indices of high-symmetry points in the k_dist array.
    save_path (str): File path for saving the figure (without extension).
    """
    import matplotlib as mpl
    # Ensure fonts are editable in Adobe Illustrator or other PDF editors
    mpl.rcParams['pdf.fonttype'] = 42

    # Initialize the figure object
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    
    # Configure spine (border) thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Configure major/minor tick styles
    ax.tick_params(which='major', length=8, width=1.5, direction='out')
    ax.tick_params(which='minor', length=4, width=1.5, direction='out')
    ax.tick_params(which='both', axis='both', right=False, top=False, bottom=True)

    # Reference Fermi level line (E-Ef = 0)
    ax.axhline(y=0.0, linestyle='--', color='black', alpha=0.5, linewidth=1.5)

    # Plot bands segment-by-segment to prevent artificial connecting lines 
    # at path discontinuities (e.g., jumps between points like H|A)
    for i in range(len(node_index) - 1):
        s0 = node_index[i]
        s1 = node_index[i+1]
        
        # Plot each band within the current segment
        for n in range(len(eigen)):
            ax.plot(k_dist[s0:s1+1], eigen[n, s0:s1+1], 
                    linewidth=2.0, color='#589fef', linestyle='solid')

    # Set horizontal axis ticks and labels
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    
    # Configure axis titles and label sizes
    ax.set_ylabel(r'${E}$-$E_{f}$ (eV)', fontsize=30)
    ax.set_xlabel('Wavevector', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    
    # Define energy axis range and ticks (from -3 to 3 eV)
    energy_ticks = np.arange(-3, 3.1, 1)
    ax.set_yticks(energy_ticks)
    ax.set_yticklabels([f"{int(t)}" if t % 1 == 0 else f"{t}" for t in energy_ticks], fontsize=25)

    # Draw vertical grid lines at high-symmetry points
    ax.grid(True, which='major', axis='x', linestyle='solid', color='gray', alpha=0.5)
    
    # Set plot boundaries
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_ylim([-3, 3])

    plt.tight_layout()
    
    # Export figures in both vector (PDF) and raster (PNG) formats
    plt.savefig(f"{save_path}.pdf", transparent=True)
    plt.savefig(f"{save_path}.png", transparent=False)
    
    # Close the figure to release memory
    plt.close()

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='band calculation')
    parser.add_argument('--config', default='band_cal.yaml', type=str, metavar='N')
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as rstream:
        input = yaml.load(rstream, yaml.SafeLoader)
    ################################ Input parameters begin ####################
    nao_max = input['nao_max']
    graph_data_path = input['graph_data_path']
    hamiltonian_path = input['hamiltonian_path']
    nk = input['nk']          # the number of k points
    save_dir = input['save_dir'] # The directory to save the results
    filename = input['strcture_name']  # The name of each cif file saved is filename_idx.cif after band calculation band from graph_data.npz

    # Ham_type
    if 'Ham_type' in input:
        Ham_type = input['Ham_type'].lower()
    else:
        Ham_type = 'openmx'
    
    # soc_switch
    if 'soc_switch' in input:
        soc_switch = input['soc_switch']
    else:
        soc_switch = False
    
    # spin_colinear
    if 'spin_colinear' in input:
        spin_colinear = input['spin_colinear']
    else:
        spin_colinear = False
    
    auto_mode = input['auto_mode']

    manual_k_info = None
    if not auto_mode:
        kpath_file = input.get('kpath_file', 'KPATH.in')
        k_path_coords, k_labels, nk_from_file = parse_kpath_file(kpath_file)
        nk = nk_from_file # replace nk from yaml
        manual_k_info = (k_path_coords, k_labels)
        
    ################################ Input parameters end ######################
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())

    num_val = np.zeros((99,), dtype=int)
    if Ham_type == 'openmx':
        for k in num_valence_openmx.keys():
            num_val[k] = num_valence_openmx[k]
    elif Ham_type == 'abacus':
        for k in num_valence_abacus.keys():
            num_val[k] = num_valence_abacus[k]
    else:
        raise NotImplementedError

    # parse the Atomic Orbital Basis Sets
    basis_definition = np.zeros((99, nao_max))
    # key is the atomic number, value is the index of the occupied orbits.
    if Ham_type == 'openmx':
        if nao_max == 14:
            basis_def = basis_def_14
        elif nao_max == 19:
            basis_def = basis_def_19
        else:
            basis_def = basis_def_26
    elif Ham_type == 'abacus':
        if nao_max == 27:
            basis_def = basis_def_27_abacus
        elif nao_max == 40:
            basis_def = basis_def_40_abacus
        else:
            raise NotImplementedError     
    else:
        raise NotImplementedError

    for k in basis_def.keys():
        basis_definition[k][basis_def[k]] = 1
    
    if soc_switch:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(2*(len(graph_dataset[i].Hon)+len(graph_dataset[i].Hoff)))
    
        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hsoc_all = []
            idx = 0
            for i in range(0, len(len_H)):
                Hsoc_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
        else:
            Hsoc_all = []
            for data in graph_dataset:
                Hsoc_all.append(torch.cat([data.Hon, data.Hoff, data.iHon, data.iHoff], dim=0).numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hsoc = Hsoc_all[idx].reshape(-1, 2*nao_max, 2*nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            cell_shift = data.cell_shift.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            cif_path = os.path.join(save_dir, filename + f'_{idx+1}.cif')
            with open(cif_path, 'w', encoding='utf-8') as f:
                f.write(struct.to(fmt="cif"))
        
            # Initialize k_path and lable        
            if auto_mode:
                kpath_seek = KPathSeek(structure = struct)
                klabels = []
                for lbs in kpath_seek.kpath['path']:
                    klabels += lbs
                # remove adjacent duplicates   
                res = [klabels[0]]
                [res.append(x) for x in klabels[1:] if x != res[-1]]
                klabels = res
                k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
                label = [rf'${lb}$' for lb in klabels]
            else:
                k_path, label = manual_k_info

            Hsoc_real, Hsoc_imag = np.split(Hsoc, 2, axis=0)
            Hsoc = [Hsoc_real[:, :nao_max, :nao_max]+1.0j*Hsoc_imag[:, :nao_max, :nao_max], 
                    Hsoc_real[:, :nao_max, nao_max:]+1.0j*Hsoc_imag[:, :nao_max, nao_max:], 
                    Hsoc_real[:, nao_max:, :nao_max]+1.0j*Hsoc_imag[:, nao_max:, :nao_max],
                    Hsoc_real[:, nao_max:, nao_max:]+1.0j*Hsoc_imag[:, nao_max:, nao_max:]]
    
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
            
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
    
            # cell index
            cell_shift_tuple = [tuple(c) for c in cell_shift.tolist()] # len: (nedges,)
            cell_shift_set = set(cell_shift_tuple)
            cell_shift_list = list(cell_shift_set)
            cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple] # len: (nedges,)
            ncells = len(cell_shift_set)
    
            # SK
            natoms = len(species)
            eigen = []
            for ik in range(nk):
                phase = np.zeros((ncells,),dtype=np.complex64) # shape (ncells,)
                phase[cell_index] = np.exp(2j*np.pi*np.sum(nbr_shift[:,:]*k_vec[ik,None,:], axis=-1))    
                na = np.arange(natoms)
        
                S_cell = np.zeros((ncells, natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                S_cell[cell_index, edge_index[0], edge_index[1], :, :] = Soff  
        
                SK = np.einsum('ijklm, i->jklm', S_cell, phase) # (natoms, natoms, nao_max, nao_max)
                SK[na,na,:,:] +=  Son[na,:,:]
                SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                SK = SK.reshape(natoms*nao_max, natoms*nao_max)
                SK = SK[orb_mask > 0]
                norbs = int(math.sqrt(SK.size))
                SK = SK.reshape(norbs, norbs)
                I = np.identity(2,dtype=np.complex64)
                SK = np.kron(I,SK)
    
                HK_list = []
                for H in Hsoc:
                    Hon = H[:natoms,:,:]
                    Hoff = H[natoms:,:,:] 
                    H_cell = np.zeros((ncells, natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                    H_cell[cell_index, edge_index[0], edge_index[1], :, :] = Hoff    
        
                    HK = np.einsum('ijklm, i->jklm', H_cell, phase) # (natoms, natoms, nao_max, nao_max)
                    HK[na,na,:,:] +=  Hon[na,:,:] # shape (nk, natoms, nao_max, nao_max)
        
                    HK = np.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
                    HK = HK.reshape(natoms*nao_max, natoms*nao_max)
        
                    # mask HK
                    HK = HK[orb_mask > 0]
                    norbs = int(math.sqrt(HK.size))
                    HK = HK.reshape(norbs, norbs)
        
                    HK_list.append(HK)
        
                HK = np.block([[HK_list[0],HK_list[1]],[HK_list[2],HK_list[3]]])
            
                SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                L = torch.linalg.cholesky(SK_cuda)
                L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                L_inv = torch.linalg.inv(L)
                L_t_inv = torch.linalg.inv(L_t)
                Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                orbital_energies, _ = torch.linalg.eigh(Hs)
                orbital_energies = orbital_energies.squeeze(0)
                eigen.append(orbital_energies.cpu().numpy())
            
            eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)
    
            # plot fermi line    
            num_electrons = np.sum(num_val[species])
            max_val = np.max(eigen[num_electrons-1])
            min_con = np.min(eigen[num_electrons])
            eigen = eigen - max_val
            print(f"max_val = {max_val} eV")
            print(f"band gap = {min_con - max_val} eV")
            
            if nk > 1:
                print('Plotting band structure...')
                save_path = os.path.join(save_dir, f'band_{idx+1}')
                plot_band_structure(k_dist, eigen, k_node, label, node_index, save_path)
                print('Done.\n')
            
            # Export energy band data
            clean_labels = []
            eigen_plot = eigen
            for lb in label:
                c_lb = lb.replace(r'$\mathrm{', '').replace('}$', '').strip()
                clean_labels.append(c_lb)
            
            with open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w") as text_file:
                text_file.write(f"# k_label: {' '.join(clean_labels)}\n")
            
                formatted_nodes = "  ".join([f"{kn:.6f}" for kn in k_node])
                text_file.write(f"# k_node: {formatted_nodes}\n")
            
                total_nk_points = len(k_dist)
                break_indices = node_index[1:] 

                for nb in range(len(eigen_plot)):
                    for ik in range(total_nk_points):
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                        
                        if ik in break_indices[:-1]:
                            text_file.write('\n')
                            text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                    text_file.write('\n')

            end_time = time.time()
            elapsed_time = end_time - start_time
        
            print("-" * 36)
            print(f"Started at: {time.ctime(start_time)}")
            print(f"Ended at: {time.ctime(end_time)}")
            print(f"Total time: {elapsed_time:.2f} s")
            print("-" * 36)

    elif spin_colinear:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(len(graph_dataset[i].Hon))
            len_H.append(len(graph_dataset[i].Hoff))

        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hon_all, Hoff_all = [], []
            idx = 0
            for i in range(0, len(len_H), 2):
                Hon_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
                Hoff_all.append(H[idx:idx + len_H[i+1]])
                idx = idx+len_H[i+1]
        else:
            Hon_all, Hoff_all = [], []
            for data in graph_dataset:
                Hon_all.append(data.Hon.numpy())
                Hoff_all.append(data.Hoff.numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hon = Hon_all[idx].reshape(-1, 2, nao_max, nao_max)
            Hoff = Hoff_all[idx].reshape(-1, 2, nao_max, nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            cif_path = os.path.join(save_dir, filename + f'_{idx+1}.cif')
            with open(cif_path, 'w', encoding='utf-8') as f:
                f.write(struct.to(fmt="cif"))
        
            # Initialize k_path and lable        
            if auto_mode:
                kpath_seek = KPathSeek(structure = struct)
                klabels = []
                for lbs in kpath_seek.kpath['path']:
                    klabels += lbs
                # remove adjacent duplicates   
                res = [klabels[0]]
                [res.append(x) for x in klabels[1:] if x != res[-1]]
                klabels = res
                k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
                label = [rf'${lb}$' for lb in klabels]        
            else:
                k_path, label = manual_k_info
                
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
        
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
        
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        
            natoms = len(struct)
            
            for ispin in range(2):
                eigen = []
                for ik in range(len(k_vec)):            
                    HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                    SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
    
                    na = np.arange(natoms)
                    HK[na,na,:,:] +=  Hon[na, ispin, :, :] # shape (natoms, nao_max, nao_max)
                    SK[na,na,:,:] +=  Son[na, :, :]
    
                    coe = np.exp(2j*np.pi*np.sum(nbr_shift*k_vec[ik][None,:], axis=-1)) # shape (nedges,)
    
                    for iedge in range(len(Hoff)):
                        # shape (nao_max, nao_max) += (1, 1)*(nao_max, nao_max)
                        HK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Hoff[iedge,ispin,:,:]
                        SK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Soff[iedge,:,:]
    
    
                    HK = np.swapaxes(HK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                    HK = HK.reshape(natoms*nao_max, natoms*nao_max)
                    SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                    SK = SK.reshape(natoms*nao_max, natoms*nao_max)
    
                    # mask HK and SK
                    #HK = torch.masked_select(HK, orb_mask[idx].repeat(nk,1,1) > 0)
                    HK = HK[orb_mask > 0]
                    norbs = int(math.sqrt(HK.size))
                    HK = HK.reshape(norbs, norbs)
    
                    #SK = torch.masked_select(SK, orb_mask[idx].repeat(nk,1,1) > 0)
                    SK = SK[orb_mask > 0]
                    norbs = int(math.sqrt(SK.size))
                    SK = SK.reshape(norbs, norbs)

                    SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                    HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                    L = torch.linalg.cholesky(SK_cuda)
                    L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                    L_inv = torch.linalg.inv(L)
                    L_t_inv = torch.linalg.inv(L_t)
                    Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                    orbital_energies, _ = torch.linalg.eigh(Hs)
                    orbital_energies = orbital_energies.squeeze(0)
                    eigen.append(orbital_energies.cpu().numpy())

                eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)

                # plot fermi line    
                num_electrons = np.sum(num_val[species])
                max_val = np.max(eigen[math.ceil(num_electrons/2)-1])
                min_con = np.min(eigen[math.ceil(num_electrons/2)])
                eigen = eigen - max_val
                print(f'band info for spin No.{ispin}')
                print(f"max_val = {max_val} eV")
                print(f"band gap = {min_con - max_val} eV")

                if nk > 1:
                    print(f'Plotting band structure...')
                    save_path = os.path.join(save_dir, f'band_spin{ispin}_{idx+1}')
                    plot_band_structure(k_dist, eigen, k_node, label, node_index, save_path)
                    print('Done.\n')

                # Export energy band data
                clean_labels = []
                eigen_plot = eigen
                for lb in label:
                    c_lb = lb.replace(r'$\mathrm{', '').replace('}$', '').strip()
                    clean_labels.append(c_lb)
            
                with open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w") as text_file:
                    text_file.write(f"# k_label: {' '.join(clean_labels)}\n")
            
                    formatted_nodes = "  ".join([f"{kn:.6f}" for kn in k_node])
                    text_file.write(f"# k_node: {formatted_nodes}\n")
            
                    total_nk_points = len(k_dist)
                    break_indices = node_index[1:] 

                    for nb in range(len(eigen_plot)):
                        for ik in range(total_nk_points):
                            text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                        
                            if ik in break_indices[:-1]:
                                text_file.write('\n')
                                text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                        text_file.write('\n')

                end_time = time.time()
                elapsed_time = end_time - start_time
        
                print("-" * 36)
                print(f"Started at: {time.ctime(start_time)}")
                print(f"Ended at: {time.ctime(end_time)}")
                print(f"Total time: {elapsed_time:.2f} s")
                print("-" * 36)
    
    else:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(len(graph_dataset[i].Hon))
            len_H.append(len(graph_dataset[i].Hoff))
               
        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hon_all, Hoff_all = [], []
            idx = 0
            for i in range(0, len(len_H), 2):
                Hon_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
                Hoff_all.append(H[idx:idx + len_H[i+1]])
                idx = idx+len_H[i+1]
        else:
            Hon_all, Hoff_all = [], []
            for data in graph_dataset:
                Hon_all.append(data.Hon.numpy())
                Hoff_all.append(data.Hoff.numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hon = Hon_all[idx].reshape(-1, nao_max, nao_max)
            Hoff = Hoff_all[idx].reshape(-1, nao_max, nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            cif_path = os.path.join(save_dir, filename + f'_{idx+1}.cif')
            with open(cif_path, 'w', encoding='utf-8') as f:
                f.write(struct.to(fmt="cif"))
        
            # Initialize k_path and lable        
            if auto_mode:
                kpath_seek = KPathSeek(structure = struct)
                klabels = []
                for lbs in kpath_seek.kpath['path']:
                    klabels += lbs
                # remove adjacent duplicates   
                res = [klabels[0]]
                [res.append(x) for x in klabels[1:] if x != res[-1]]
                klabels = res
                k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
                label = [rf'${lb}$' for lb in klabels]     
            else:
                k_path, label = manual_k_info
        
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
        
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
        
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        
            natoms = len(struct)
            eigen = []
            for ik in range(nk):
                HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
            
                na = np.arange(natoms)
                HK[na,na,:,:] +=  Hon[na,:,:] # shape (natoms, nao_max, nao_max)
                SK[na,na,:,:] +=  Son[na,:,:]
            
                coe = np.exp(2j*np.pi*np.sum(nbr_shift*k_vec[ik][None,:], axis=-1)) # shape (nedges,)
            
                for iedge in range(len(Hoff)):
                    # shape (nao_max, nao_max) += (1, 1)*(nao_max, nao_max)
                    HK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Hoff[iedge,:,:]
                    SK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Soff[iedge,:,:]
            
            
                HK = np.swapaxes(HK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                HK = HK.reshape(natoms*nao_max, natoms*nao_max)
                SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                SK = SK.reshape(natoms*nao_max, natoms*nao_max)
            
                # mask HK and SK
                #HK = torch.masked_select(HK, orb_mask[idx].repeat(nk,1,1) > 0)
                HK = HK[orb_mask > 0]
                norbs = int(math.sqrt(HK.size))
                HK = HK.reshape(norbs, norbs)
                        
                #SK = torch.masked_select(SK, orb_mask[idx].repeat(nk,1,1) > 0)
                SK = SK[orb_mask > 0]
                norbs = int(math.sqrt(SK.size))
                SK = SK.reshape(norbs, norbs)

                SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                L = torch.linalg.cholesky(SK_cuda)
                L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                L_inv = torch.linalg.inv(L)
                L_t_inv = torch.linalg.inv(L_t)
                Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                orbital_energies, _ = torch.linalg.eigh(Hs)
                orbital_energies = orbital_energies.squeeze(0)
                eigen.append(orbital_energies.cpu().numpy())
            
            eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)
            
            # plot fermi line    
            num_electrons = np.sum(num_val[species])
            max_val = np.max(eigen[math.ceil(num_electrons/2)-1])
            min_con = np.min(eigen[math.ceil(num_electrons/2)])
            eigen = eigen - max_val
            print(f"max_val = {max_val} eV")
            print(f"band gap = {min_con - max_val} eV")
            
            if nk > 1:
                    print('Plotting band structure ...')
                    save_path = os.path.join(save_dir, f'band_{idx+1}')
                    plot_band_structure(k_dist, eigen, k_node, label, node_index, save_path)
                    print('Done.\n')
            
            # Export energy band data
            clean_labels = []
            eigen_plot = eigen
            for lb in label:
                c_lb = lb.replace(r'$\mathrm{', '').replace('}$', '').strip()
                clean_labels.append(c_lb)
            
            with open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w") as text_file:
                text_file.write(f"# k_label: {' '.join(clean_labels)}\n")
            
                formatted_nodes = "  ".join([f"{kn:.6f}" for kn in k_node])
                text_file.write(f"# k_node: {formatted_nodes}\n")
            
                total_nk_points = len(k_dist)
                break_indices = node_index[1:] 

                for nb in range(len(eigen_plot)):
                    for ik in range(total_nk_points):
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                        
                        if ik in break_indices[:-1]:
                            text_file.write('\n')
                            text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                    text_file.write('\n')

            end_time = time.time()
            elapsed_time = end_time - start_time
        
            print("-" * 36)
            print(f"Started at: {time.ctime(start_time)}")
            print(f"Ended at: {time.ctime(end_time)}")
            print(f"Total time: {elapsed_time:.2f} s")
            print("-" * 36)

if __name__ == '__main__':
    main()

