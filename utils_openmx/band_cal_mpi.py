'''
Descripttion: The script to calculat bands from the results of HamGNN
version: 1.0
Author: Yang Zhong
Date: 2022-12-20 14:08:52
LastEditors: Hao-Jen You
LastEditTime: 2026-01-27 12:00:00
'''

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.core.periodic_table import Element
import math
import os, sys
from utils_openmx.utils import *
import argparse
import yaml
import torch
import time
from mpi4py import MPI

torch.set_num_threads(1)

def get_fermi_weight(energy_diff, kt=0.01):
    """
    Calculate the Fermi-Dirac distribution.
    
    Parameters:
    energy_diff : float or ndarray
        The energy difference (E - Ef) in eV.
    kt : float
        The thermal smearing width (k_B * T) in eV.
    
    Returns:
    float or ndarray
        The occupancy weights (0.0 to 1.0).
    """
    arg = energy_diff / kt
    # Clip arg range to prevent numerical overflow in np.exp
    return 1.0 / (np.exp(np.clip(arg, -60, 60)) + 1.0)

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

def plot_band_structure(k_dist, eigen, k_node, label, node_index, save_path, n_occ):
    """Plot band structure using professional publication-quality template."""
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.tick_params(which='major', length=8, width=1.5, direction='out')
    ax.tick_params(which='minor', length=4, width=1.5, direction='out')
    ax.tick_params(which='both', axis='both', right=False, top=False, bottom=True)
    ax.axhline(y=0.0, linestyle='--', color='black', alpha=0.5, linewidth=1.5)

    color_vb = '#589fef' # blue
    color_cb = '#56c278' # green

    for n in range(len(eigen)):
        if n < n_occ:
            band_color = color_vb
            label_name = 'Valence'
        else:
            band_color = color_cb
            label_name = 'Conduction'
        
        for i in range(len(node_index) - 1):
            s0 = node_index[i]
            s1 = node_index[i+1]
            line_label = label_name if (n == 0 or n == n_occ) and i == 0 else None
            
            ax.plot(k_dist[s0:s1+1], eigen[n, s0:s1+1], 
                    linewidth=2.0, color=band_color, label=line_label)

    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    ax.set_ylabel(r'${E}$-$E_{f}$ (eV)', fontsize=30)
    ax.set_xlabel('Wavevector', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    
    energy_ticks = np.arange(-3, 3.1, 1)
    ax.set_yticks(energy_ticks)
    ax.set_yticklabels([f"{int(t)}" if t % 1 == 0 else f"{t}" for t in energy_ticks], fontsize=25)
    ax.grid(True, which='major', axis='x', linestyle='solid', color='gray', alpha=0.5)
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_ylim([-3, 3])
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", transparent=True)
    plt.savefig(f"{save_path}.png", transparent=False)
    plt.close()

def main():
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    parser = argparse.ArgumentParser(description='MPI Band Calculation')
    parser.add_argument('--config', default='band_cal.yaml', type=str)
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as rstream:
        input_cfg = yaml.load(rstream, yaml.SafeLoader)

    nao_max = input_cfg['nao_max']
    graph_data_path = input_cfg['graph_data_path']
    hamiltonian_path = input_cfg.get('hamiltonian_path', None)
    nk = input_cfg.get('nk', 20) 
    save_dir = input_cfg['save_dir']
    filename = input_cfg['strcture_name']
    Ham_type = input_cfg.get('Ham_type', 'openmx').lower()
    soc_switch = input_cfg.get('soc_switch', False)
    spin_colinear = input_cfg.get('spin_colinear', False)
    auto_mode = input_cfg['auto_mode']

    k_path_coords, k_labels = None, None
    if not auto_mode:
        kpath_file = input_cfg.get('kpath_file', 'KPATH.in')
        k_path_coords, k_labels, nk_from_file = parse_kpath_file(kpath_file)
        nk = nk_from_file 

    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    graph_data = np.load(graph_data_path, allow_pickle=True)['graph'].item()
    graph_dataset = list(graph_data.values())

    num_val = np.zeros((99,), dtype=int)
    val_source = num_valence_openmx if Ham_type == 'openmx' else num_valence_abacus
    for k, v in val_source.items(): num_val[k] = v

    basis_definition = np.zeros((99, nao_max))
    if Ham_type == 'openmx':
        basis_def = {14: basis_def_14, 19: basis_def_19}.get(nao_max, basis_def_26)
    else:
        basis_def = {27: basis_def_27_abacus, 40: basis_def_40_abacus}.get(nao_max)
    
    for k, v in basis_def.items(): basis_definition[k][v] = 1

    Hsoc_all, Hon_all, Hoff_all = [], [], []
    if soc_switch:
        if hamiltonian_path:
            H_raw = np.load(hamiltonian_path)
            idx_tmp = 0
            for data in graph_dataset:
                length = 2 * (len(data.Hon) + len(data.Hoff))
                Hsoc_all.append(H_raw[idx_tmp:idx_tmp + length]); idx_tmp += length
        else:
            for data in graph_dataset:
                Hsoc_all.append(torch.cat([data.Hon, data.Hoff, data.iHon, data.iHoff], dim=0).numpy())
    else:
        if hamiltonian_path:
            H_raw = np.load(hamiltonian_path)
            idx_tmp = 0
            for data in graph_dataset:
                l_on, l_off = len(data.Hon), len(data.Hoff)
                Hon_all.append(H_raw[idx_tmp:idx_tmp+l_on]); idx_tmp += l_on
                Hoff_all.append(H_raw[idx_tmp:idx_tmp+l_off]); idx_tmp += l_off
        else:
            for data in graph_dataset:
                Hon_all.append(data.Hon.numpy()); Hoff_all.append(data.Hoff.numpy())

    for idx, data in enumerate(graph_dataset):
        latt = data.cell.numpy().reshape(3,3)
        species = data.z.numpy()
        pos = data.pos.numpy() * au2ang
        nbr_shift = data.nbr_shift.numpy()
        edge_index = data.edge_index.numpy()
        Son, Soff = data.Son.numpy().reshape(-1, nao_max, nao_max), data.Soff.numpy().reshape(-1, nao_max, nao_max)
        natoms = len(species)

        if rank == 0:
            struct = Structure(latt * au2ang, [Element.from_Z(k).symbol for k in species], pos, coords_are_cartesian=True)
            with open(os.path.join(save_dir, f"{filename}_{idx+1}.cif"), 'w') as f: f.write(struct.to(fmt="cif"))
            
            if auto_mode:
                kpath_seek = KPathSeek(structure=struct)
                klabels = []
                for lbs in kpath_seek.kpath['path']: klabels += lbs
                res = [klabels[0]]
                [res.append(x) for x in klabels[1:] if x != res[-1]]
                current_k_path = [kpath_seek.kpath['kpoints'][k] for k in res]
                current_label = [rf'$\mathrm{{{lb.replace("GAMMA","Γ")}}}$' for lb in res]
            else:
                current_k_path, current_label = k_path_coords, k_labels
        else:
            current_k_path, current_label = None, None

        current_k_path = comm.bcast(current_k_path, root=0)
        current_label = comm.bcast(current_label, root=0)

        kpts_gen = kpoints_generator(dim_k=3, lat=latt)
        k_vec_all, k_dist, k_node, lat_per_inv, node_index = kpts_gen.k_path(current_k_path, nk)
        k_vec_all = k_vec_all.dot(lat_per_inv[np.newaxis,:,:]).reshape(-1,3)
        orb_mask = (basis_definition[species].reshape(-1)[:,None] * basis_definition[species].reshape(-1)[None,:]).astype(bool)
        my_indices = np.array_split(np.arange(len(k_vec_all)), size)[rank]

        eigen_local = []
        if soc_switch:
            Hsoc = Hsoc_all[idx].reshape(-1, 2*nao_max, 2*nao_max)
            Hr, Hi = np.split(Hsoc, 2, axis=0)
            H_blocks = [Hr[:,:nao_max,:nao_max] + 1j*Hi[:,:nao_max,:nao_max],
                        Hr[:,:nao_max,nao_max:] + 1j*Hi[:,:nao_max,nao_max:],
                        Hr[:,nao_max:,:nao_max] + 1j*Hi[:,nao_max:,:nao_max],
                        Hr[:,nao_max:,nao_max:] + 1j*Hi[:,nao_max:,nao_max:]]
            for ik in my_indices:
                phase = np.exp(2j * np.pi * np.sum(nbr_shift * k_vec_all[ik], axis=-1))
                SK_f = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                np.add.at(SK_f, (edge_index[0], edge_index[1]), Soff * phase[:, None, None])
                for i in range(natoms): SK_f[i, i] += Son[i]
                SK_mat = np.swapaxes(SK_f, 1, 2).reshape(natoms*nao_max, natoms*nao_max)[orb_mask]
                norbs = int(math.sqrt(SK_mat.size))
                SK_final = np.kron(np.eye(2, dtype=np.complex64), SK_mat.reshape(norbs, norbs))
                HK_list = []
                for Hb in H_blocks:
                    HK_sub = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                    np.add.at(HK_sub, (edge_index[0], edge_index[1]), Hb[natoms:] * phase[:, None, None])
                    for i in range(natoms): HK_sub[i, i] += Hb[i]
                    HK_list.append(np.swapaxes(HK_sub, 1, 2).reshape(natoms*nao_max, natoms*nao_max)[orb_mask].reshape(norbs, norbs))
                HK_final = np.block([[HK_list[0], HK_list[1]], [HK_list[2], HK_list[3]]])
                L = torch.linalg.cholesky(torch.from_numpy(SK_final))
                L_inv = torch.linalg.inv(L)
                Hs = L_inv @ torch.from_numpy(HK_final) @ L_inv.conj().transpose(-2, -1)
                eigen_local.append(torch.linalg.eigvalsh(Hs).numpy())
        else:
            Hon, Hoff = Hon_all[idx].reshape(-1, nao_max, nao_max), Hoff_all[idx].reshape(-1, nao_max, nao_max)
            for ik in my_indices:
                phase = np.exp(2j * np.pi * np.sum(nbr_shift * k_vec_all[ik], axis=-1))
                HK_f, SK_f = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64), np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                np.add.at(HK_f, (edge_index[0], edge_index[1]), Hoff * phase[:, None, None])
                np.add.at(SK_f, (edge_index[0], edge_index[1]), Soff * phase[:, None, None])
                for i in range(natoms): HK_f[i,i] += Hon[i]; SK_f[i,i] += Son[i]
                HK = np.swapaxes(HK_f, 1, 2).reshape(natoms*nao_max, natoms*nao_max)[orb_mask]
                SK = np.swapaxes(SK_f, 1, 2).reshape(natoms*nao_max, natoms*nao_max)[orb_mask]
                norbs = int(math.sqrt(HK.size))
                L = torch.linalg.cholesky(torch.from_numpy(SK.reshape(norbs, norbs)))
                L_inv = torch.linalg.inv(L)
                Hs = L_inv @ torch.from_numpy(HK.reshape(norbs, norbs)) @ L_inv.conj().transpose(-2, -1)
                eigen_local.append(torch.linalg.eigvalsh(Hs).numpy())

        comm.Barrier()
        gathered_eigen = comm.gather(eigen_local, root=0)

        if rank == 0:
            # Flatten and transpose eigenvalues from all ranks, then convert from Atomic Units (Hartree) to eV
            all_eigen_raw = np.array([e for r in gathered_eigen for e in r]).T * au2ev
            num_electrons = np.sum(num_val[species])
            
            # --- 1. Fermi Level Calculation ---
            # Determine target electron count; if SOC is off, divide by 2 for spin degeneracy
            target_nelec = num_electrons if soc_switch else num_electrons / 2.0
            elw, eup = np.min(all_eigen_raw) - 1.0, np.max(all_eigen_raw) + 1.0
            
            # Solve for Fermi Level using Bisection Method
            for i in range(100): 
                ef_test = (elw + eup) / 2.0
                sigma = 0.01  # Smearing width in eV
                occupancy = get_fermi_weight(all_eigen_raw - ef_test, kt=sigma)
                total_n = np.sum(occupancy) / len(k_dist)
                
                if total_n < target_nelec:
                    elw = ef_test
                else:
                    eup = ef_test
            
            e_fermi = (elw + eup) / 2.0

            # Shift all eigenvalues relative to the calculated Fermi Level
            eigen_plot = all_eigen_raw - e_fermi

            # --- 2. Identify Occupied Band Indices ---
            n_occ = int(round(target_nelec))
            # vb_band (Valence Band) is the n_occ-th band (index n_occ-1)
            # cb_band (Conduction Band) is the (n_occ+1)-th band (index n_occ)
            vb_band = eigen_plot[n_occ - 1, :]
            cb_band = eigen_plot[n_occ, :]

            vbm = np.max(vb_band)
            cbm = np.min(cb_band)
            gap = cbm - vbm

            # Find the indices for VBM/CBM along the K-path
            vbm_k_idx = np.argmax(vb_band)
            cbm_k_idx = np.argmin(cb_band)

            # --- 3. Auto-detect System Type and Printing ---
            print("-" * 53)
            print(f"Calculated Fermi Level: {e_fermi:.6f} eV")
            vbm_k_coords = k_vec_all[vbm_k_idx]
            cbm_k_coords = k_vec_all[cbm_k_idx]
            gap_type = "Direct" if vbm_k_idx == cbm_k_idx else "Indirect"
            print(f"        Band Character:    {gap_type}")
            print(f"         Band Gap (eV):    {gap:.4f}")
            print(f"Eigenvalue of VBM (eV):    {vbm+e_fermi:.4f}")
            print(f"Eigenvalue of CBM (eV):    {cbm+e_fermi:.4f}")
            print(f"     HOMO & LUMO Bands:        {n_occ}        {n_occ + 1}")
            print(f"       Location of VBM:  {vbm_k_coords[0]:.6f}  {vbm_k_coords[1]:.6f}  {vbm_k_coords[2]:.6f}")
            print(f"       Location of CBM:  {cbm_k_coords[0]:.6f}  {cbm_k_coords[1]:.6f}  {cbm_k_coords[2]:.6f}")
            print("-" * 53)
            #print('Plotting band structure ...')
            # Pass n_occ to the plotting function for band color differentiation
            plot_band_structure(k_dist, eigen_plot, k_node, current_label, node_index, 
                                os.path.join(save_dir, f'band_{idx+1}'), n_occ)
            
            # --- 4. Exporting Data to .dat File ---
            # Remove LaTeX formatting from labels for raw text export
            clean_labels = [lb.replace(r'$\mathrm{', '').replace('}$', '').strip() for lb in current_label]
            with open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w") as text_file:
                text_file.write(f"# k_label: {' '.join(clean_labels)}\n")
                text_file.write(f"# k_node: {' '.join([f'{kn:.6f}' for kn in k_node])}\n")
                
                break_indices = node_index[1:]
                for nb in range(len(eigen_plot)):
                    for ik in range(len(k_dist)):
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                        
                        # Handle discontinuities in the K-path (high-symmetry points)
                        if ik in break_indices[:-1]:
                            text_file.write('\n')
                            text_file.write("%f    %f\n" % (k_dist[ik], eigen_plot[nb, ik]))
                    # Add extra newline between bands for Gnuplot/standard compatibility
                    text_file.write('\n')

            end_time = time.time()
            elapsed_time = end_time - start_time
        
            print(f"Started at: {time.ctime(start_time)}")
            print(f"Ended at: {time.ctime(end_time)}")
            print(f"Total time: {elapsed_time:.2f} s")
            print("-" * 53)

if __name__ == '__main__':
    main()