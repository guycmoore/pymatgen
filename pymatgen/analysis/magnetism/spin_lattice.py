# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
"""

import numpy as np
import numpy.linalg as npla
import scipy as sp
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix

import sparseqr
import random

from pymatgen.core import Structure, Lattice, Element
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.io.cif import CifWriter

from TB2J.io_exchange import SpinIO

__author__ = "Guy C. Moore"
__version__ = "0.0"
__maintainer__ = "Guy C. Moore"
__email__ = "gmoore@lbl.gov"
__status__ = "Development"
__date__ = "March 2021"

def get_clusters_pair(adj_matrix_tri, num_site):

    clusters = []
    for i in range(num_site):
        for link in adj_matrix_tri[i]:
            j = link['id']
            img = link['to_jimage']
            # forward cluster
            cluster = (i,j,img)
            clusters.append(cluster)
            # backward cluster
            cluster = (j,i,tuple([-d for d in img]))
            clusters.append(cluster)

    return clusters

def get_intersite_uvecs(structure, clusters, normalize=True):

    # unit vectors between sites
    uvecs = {}
    for cluster in clusters:
        (i, j, to_jimage) = cluster
        # cartesian coordinates
        svec = np.dot(structure.lattice.matrix.T, to_jimage)
        vvec = structure[j].coords - structure[i].coords + svec
        # # lattice coordinates
        # vvec = structure[j].frac_coords - structure[i].frac_coords + to_jimage
        # normalize vector
        if normalize:
            vvec = vvec / npla.norm(vvec)
        uvecs[cluster] = vvec
        
        # print(to_jimage, vvec, npla.norm(vvec))
        
    return uvecs

def get_configurations(
    struct, num_config_mag, num_config_nonmag, disp_std=None):

    if not disp_std:
        disp_std = 0.01 * (npla.det(self.struct.lattice.matrix) / len(struct))**(1/3)

    def rand_spin_unif(r):
        
        pos = np.zeros([3])
        while npla.norm(pos) == 0.0:
            pos = np.random.normal(0.0, 1.0, [3])
        pos /= npla.norm(pos)
        
        # theta = 2*np.pi * random.uniform(0.0, 1.0)
        # phi = np.arccos(random.uniform(-1.0, 1.0))
        # pos = np.zeros([3])
        # pos[0] = r*np.sin(phi)*np.cos(theta)
        # pos[1] = r*np.sin(phi)*np.sin(theta)
        # pos[2] = r*np.cos(phi)
        
        return pos

    def rand_disp_gauss(sigma):
        pos = np.random.normal(0.0,sigma,[3])
        return pos

    states_magn = [[rand_spin_unif(npla.norm(list(s.properties['magmom']))) for s in struct]
                    for i in range(num_config_mag)]
    states_disp = [[rand_disp_gauss(disp_std) for s in struct]
                    for i in range(num_config_nonmag)]

    return states_magn, states_disp

class SpinDisplaceBondProjected:
    def __init__(self, structure, magmom_tol=0.1):
        
        self.magmom_tol = magmom_tol

        # create magnetic structure 
        sites_mag, sites_nonmag = [], []
        for site in structure:
            if npla.norm(list(site.properties['magmom'])) > self.magmom_tol:
                sites_mag.append(site)
            else:
                sites_nonmag.append(site)
        self.struct_mag = Structure.from_sites(sites=sites_mag)
        self.magn_species = [str(s) for s in self.struct_mag.species]

        self.struct = Structure.from_sites(sites=sites_mag+sites_nonmag)

        # system parameters
        self.num_spin = len(self.struct_mag)
        self.num_site = len(self.struct)
        self.num_config_mag, self.num_config_nonmag = -1, -1

        # mapping from magnetic sites to parent structure
        self.magn_struct_map = [self.struct.index(site_mag) for site_mag in self.struct_mag]

        strategy = MinimumDistanceNN() # MinimumDistanceNN(cutoff=cutoff, get_all_sites=False)
        graph_mag = StructureGraph.with_local_env_strategy(self.struct_mag, strategy=strategy)
        self.adj_matrix_tri_mag = graph_mag.as_dict()['graphs']['adjacency']

        self.adj_matrix_mag = [[] for i in range(self.num_spin)]
        for i in range(self.num_spin):
            for link in self.adj_matrix_tri_mag[i]:
                j = link['id']
                self.adj_matrix_mag[i].append(j)
                self.adj_matrix_mag[j].append(i)
        self.adj_matrix_mag = [list(set(a)) for a in self.adj_matrix_mag]

        strategy = MinimumDistanceNN() # MinimumDistanceNN(cutoff=cutoff, get_all_sites=False)
        graph = StructureGraph.with_local_env_strategy(self.struct, strategy=strategy)
        self.adj_matrix_tri = graph.as_dict()['graphs']['adjacency']

        self.adj_matrix = [[] for i in range(self.num_site)]
        for i in range(self.num_site):
            for link in self.adj_matrix_tri[i]:
                j = link['id']
                self.adj_matrix[i].append(j)
                self.adj_matrix[j].append(i)
        self.adj_matrix = [list(set(a)) for a in self.adj_matrix]

        # setup indices
        self.enumerate_cluster_indices()

        # unit vectors between sites
        self.uvecs = get_intersite_uvecs(self.struct, self.unique_pairs_all, normalize=True)
        self.vvecs = get_intersite_uvecs(self.struct, self.unique_pairs_all, normalize=False)

        struct_nonmag = Structure(species=self.struct.species, 
                coords=self.struct.frac_coords, 
                lattice=self.struct.lattice)

        #self.sga = SpacegroupAnalyzer(structure=struct_nonmag, symprec=0.01, angle_tolerance=5.0)
        self.sga = SpacegroupAnalyzer(structure=struct_nonmag, symprec=0.05, angle_tolerance=5.0)
        print("Spacegroup: ", self.sga.get_space_group_symbol())

        self.symmops = self.sga.get_space_group_operations()

        # setup matrices
        self.construct_matrices()

#     def test_bond_projected(self, 
#                             num_config,
#                             j_matrix, jp_matrix, k_matrix):

#         xorig = np.zeros([self.num_params])
#         n = -1
#         for cluster in self.unique_pairs_mag:
#             n += 1
#             xorig[n] = j_matrix[cluster[0], cluster[1]]
#         for cluster in self.unique_pairs:
#             n += 1
#             xorig[n] = k_matrix[cluster[0],cluster[1]]
#         for cluster in self.unique_pairs_mag:
#             n += 1
#             xorig[n] = jp_matrix[cluster[0],cluster[1]]

#         # construct data matrix
#         self.states_magn, self.states_disp = self.get_configurations(
#             num_config, num_config, disp_std=None)

#         states_magn = self.states_magn
#         states_disp_mag, states_disp_nonmag = self.states_disp, self.states_disp

#         forces_mag    = [list(np.zeros([len(self.struct), 3])) for i in range(num_config)]
#         forces_nonmag = [list(np.zeros([len(self.struct), 3])) for i in range(num_config)]

#         hmags         = [list(np.zeros([len(self.struct), 3])) for i in range(num_config)]

#         energies_mag    = [0.0 for i in range(num_config)]
#         energies_nonmag = [0.0 for i in range(num_config)]        

#         # self.construct_data_matrix_nonmagnetic(
#         #     num_config,
#         #     states_disp_nonmag, energies_nonmag, forces_nonmag)
#         self.construct_data_matrix_magnetic(
#             num_config,
#             states_magn, states_disp_mag,
#             energies_mag, forces_mag, hmags)
#         self.construct_data_matrix()

#         self.datavec = self.config_matrix.dot(xorig)

#         return xorig

    def perform_fitting (self):

        self.setup_constrained_least_squares()
        self.solve_constrained_least_squares()

    def get_model_parameters(self):

        j_matrix, jp_matrix, k_para_matrix, k_perp_matrix = {}, {}, {}, {}

        xcomp_t = self.xcomp.T[0]

        n = -1
        for cluster in self.unique_pairs_mag:
            n += 1
            j_matrix[cluster] = xcomp_t[n]
        for cluster in self.unique_pairs_mag:
            n += 1
            jp_matrix[cluster] = xcomp_t[n]
        for cluster in self.unique_pairs:
            n += 1
            k_para_matrix[cluster] = xcomp_t[n]
        for cluster in self.unique_pairs:
            n += 1
            k_perp_matrix[cluster] = xcomp_t[n]

        return j_matrix, jp_matrix, k_para_matrix, k_perp_matrix

    def enumerate_cluster_indices(self):
        
        self.unique_pairs_mag = get_clusters_pair(self.adj_matrix_tri_mag, self.num_spin)
        self.unique_pairs = get_clusters_pair(self.adj_matrix_tri, self.num_site)
        
        # all unique pairs
        self.unique_pairs_all = list(set(self.unique_pairs+self.unique_pairs_mag))
        
        self.unique_triads_ijk = []
        for cluster_a in self.unique_pairs_mag:
            for cluster_b in self.unique_pairs:
                for cluster_c in self.unique_pairs:
                    idxs = set(list(cluster_a[0:2])+list(cluster_b[0:2])+list(cluster_c[0:2]))
                    if (len(idxs) == 3) and (cluster_b[0] == cluster_c[0]):
                        cluster = (cluster_b[1], cluster_c[1], cluster_b[0], 
                                   (cluster_b[-1], cluster_c[-1]))
                        self.unique_triads_ijk.append(cluster)
        self.unique_triads_ijk = list(set(self.unique_triads_ijk))
        
        # # include both NN and NN in magnetic structure
        # self.unique_pairs = list(set(self.unique_pairs+self.unique_pairs_mag))
        # # # only examine NN (magnetic)
        # # self.unique_pairs = self.unique_pairs_mag
        
        self.num_params = 2 + 3*len(self.unique_pairs_mag) + len(self.unique_triads_ijk) + 7*len(self.unique_pairs)
        
        print("Number of unique pairs = ", len(self.unique_pairs))
        print("Number of unique pairs (magnetic) = ", len(self.unique_pairs_mag))
        print("Number of unique triads (ijk) = ", len(self.unique_triads_ijk))
        print()
        
    def setup_constrained_least_squares(self):

        self.q_c, self.r_c, self.p_c, self.rank_c = sparseqr.qr(self.a_constr.T);

        self.uq = coo_matrix.dot(self.config_matrix, self.q_c)
        self.q_p, self.r_p = sp.linalg.qr(self.uq[0:, self.rank_c:].todense(), pivoting=False)

    def solve_constrained_least_squares(self):

        # Try: Tikhonov or Lasso
        self.z0 = self.q_p.T.dot(self.datavec)
        self.z = sp.sparse.linalg.lsqr(self.r_p, self.z0)[0]
        self.y = np.vstack((np.zeros([self.rank_c, 1]), np.transpose([self.z])))
        self.xcomp = self.q_c.dot(self.y)

    def construct_matrices(self):

        self.get_full_constraint_matrix()

    def construct_data_matrix(self):

        vals, rows, cols = [], [], []
        datavec = []
        row_shift = 0

        # magnetic data
        rows.extend(row_shift+self.config_matrix_magnetic.row)
        cols.extend(self.config_matrix_magnetic.col)
        vals.extend(self.config_matrix_magnetic.data)
        datavec.extend(self.datavec_magnetic)
        row_shift += self.config_matrix_magnetic.shape[0]

        # nonmagnetic data
        rows.extend(row_shift+self.config_matrix_nonmagnetic.row)
        cols.extend(self.config_matrix_nonmagnetic.col)
        vals.extend(self.config_matrix_nonmagnetic.data)
        datavec.extend(self.datavec_nonmagnetic)
        row_shift += self.config_matrix_nonmagnetic.shape[0]

        # construct full system
        row_len = row_shift
        col_len = self.num_params
        self.config_matrix = coo_matrix(
            (vals, (rows, cols)), shape=(row_len, col_len), dtype=float)
        self.datavec = datavec.copy()

    def construct_data_matrix_from_input(self, 
            num_config_mag,
            is_magnetic=True,
            states_magn=None, states_disp=None,
            energies=None, forces=None, hmags=None,
            j_matrix_input=None, fit_anharmonic=False):
        
        self.num_config_mag = num_config_mag
        
        self.states_magn, self.states_disp = states_magn, states_disp
        self.energies, self.forces, self.hmags = energies, forces, hmags
        
        if is_magnetic:
            # normalize moments
            for i in range(self.num_config_mag):
                for j in range(len(self.states_magn[i])):
                    m = self.states_magn[i][j]
                    if npla.norm(m) > self.magmom_tol:
                        self.states_magn[i][j] = np.array(m) / npla.norm(m)
                    else:
                        self.states_magn[i][j] = 0.0*np.array(m)
        
        ####################################
        # Cluster functions
        
        ###############
        # Energies
        
        def calc_ss(cluster, i, toggle=1):
            ss = np.dot(
                self.states_magn[i][cluster[0]], 
                self.states_magn[i][cluster[1]])
            return toggle*ss
        
        def calc_ssu_para(cluster, i, toggle=1):
            ss = np.dot(
                self.states_magn[i][cluster[0]], 
                self.states_magn[i][cluster[1]])
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            ssu_para = ss * u_para
            return toggle*ssu_para
        
        # def calc_ssu_ijk(cluster, i, toggle=1):
        #     s_ip = np.dot(self.states_magn[i][cluster[0]], self.states_magn[i][cluster[1]])
        #     cluster_ik = (cluster[2], cluster[0], cluster[-1][0])
        #     cluster_jk = (cluster[2], cluster[1], cluster[-1][1])
        #     u_diff_ik = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[2]]
        #     u_diff_jk = self.states_disp[i][cluster[1]] - self.states_disp[i][cluster[2]]
        #     ssu_ijk = s_ip * (np.dot(u_diff_ik, self.uvecs[cluster_ik]) + np.dot(u_diff_jk, self.uvecs[cluster_jk]))
        #     return toggle*ssu_ijk
        
        def calc_ssu_ijk(cluster, i, toggle=1):
            s_ip = np.dot(self.states_magn[i][cluster[0]], self.states_magn[i][cluster[1]])
            cluster_ik = (cluster[2], cluster[0], cluster[-1][0])
            cluster_jk = (cluster[2], cluster[1], cluster[-1][1])
            ssu_ijk = s_ip * np.dot(
                self.states_disp[i][cluster[2]], self.uvecs[cluster_ik] + self.uvecs[cluster_jk])
            return toggle*ssu_ijk
        
        def calc_uu_para(cluster, i, toggle=1):
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            uu_para = u_para**2
            return toggle*uu_para
        
        def calc_uu_perp(cluster, i, toggle=1):
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            uu_perp = npla.norm(u_diff)**2 - u_para**2
            return toggle*uu_perp
        
        ###############
        # Derivatives
        
        def calc_ss_deriv_s(cluster, k, kl, i, toggle=1):
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            s_ix, s_jx = self.states_magn[i][cluster[0]][kl], self.states_magn[i][cluster[1]][kl]
            ss_deriv_s = s_jx*del_ik + s_ix*del_jk
            return toggle*ss_deriv_s
        
        def calc_ssu_para_deriv_s(cluster, k, kl, i, toggle=1):
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            s_ix, s_jx = self.states_magn[i][cluster[0]][kl], self.states_magn[i][cluster[1]][kl]
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            ssu_para_deriv_s = u_para*(s_jx*del_ik + s_ix*del_jk)
            return toggle*ssu_para_deriv_s
        
        def calc_ssu_para_deriv_u(cluster, k, kl, i, toggle=1):
            s_ip = np.dot(self.states_magn[i][cluster[0]], self.states_magn[i][cluster[1]])
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            u_proj_p = (del_ik - del_jk) * self.uvecs[cluster][kl]
            ssu_para_deriv_u = s_ip * u_proj_p
            return toggle*ssu_para_deriv_u
        
        # def calc_ssu_ijk_deriv_u(cluster, k, kl, i, toggle=1):
        #     s_ip = np.dot(self.states_magn[i][cluster[0]], self.states_magn[i][cluster[1]])
        #     cluster_ik = (cluster[2], cluster[0], cluster[-1][0])
        #     cluster_jk = (cluster[2], cluster[1], cluster[-1][1])
        #     #
        #     if k == cluster[2]:
        #         ssu_ijk_deriv_u = s_ip * (self.uvecs[cluster_ik][kl] + self.uvecs[cluster_jk][kl])
        #     elif k == cluster[1]:
        #         ssu_ijk_deriv_u = - s_ip * self.uvecs[cluster_jk][kl]
        #     elif k == cluster[0]:
        #         ssu_ijk_deriv_u = - s_ip * self.uvecs[cluster_ik][kl]
        #     else:
        #         ssu_ijk_deriv_u = 0.0
        #     #
        #     return toggle*ssu_ijk_deriv_u
        
        def calc_ssu_ijk_deriv_u(cluster, k, kl, i, toggle=1):
            s_ip = np.dot(self.states_magn[i][cluster[0]], self.states_magn[i][cluster[1]])
            cluster_ik = (cluster[2], cluster[0], cluster[-1][0])
            cluster_jk = (cluster[2], cluster[1], cluster[-1][1])
            #
            if k == cluster[2]:
                ssu_ijk_deriv_u = s_ip * (self.uvecs[cluster_ik][kl] + self.uvecs[cluster_jk][kl])
            else:
                ssu_ijk_deriv_u = 0.0
            #
            return toggle*ssu_ijk_deriv_u
        
        # # HACK: Spring test
        # def calc_uu_para_deriv_u(cluster, k, kl, i, toggle=1):
        #     del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
        #     u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
        #     dist_frac = npla.norm(self.vvecs[cluster]) / npla.norm(self.vvecs[cluster] - u_diff)
        #     rp_deriv_u = self.vvecs[cluster][kl] - u_diff[kl]
        #     uu_para_deriv_u = - 2.0 * (1.0 - dist_frac) * (del_ik - del_jk) * rp_deriv_u
        #     # print(dist_frac, npla.norm(self.vvecs[cluster]))
        #     return toggle*uu_para_deriv_u
        
        def calc_uu_para_deriv_u(cluster, k, kl, i, toggle=1):
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            u_para_deriv_u = (del_ik - del_jk) * self.uvecs[cluster][kl]
            uu_para_deriv_u = 2.0 * u_para * u_para_deriv_u
            return toggle*uu_para_deriv_u
        
        def calc_uu_perp_deriv_u(cluster, k, kl, i, toggle=1):
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            uu_diff_deriv_u = 2.0 * (del_ik - del_jk) * u_diff[kl]
            uu_para_deriv_u = calc_uu_para_deriv_u(cluster, k, kl, i)
            uu_perp_deriv_u = uu_diff_deriv_u - uu_para_deriv_u
            return toggle*uu_perp_deriv_u
        
        ########
        # Anharmonic terms
        
        def calc_ssuu_perp_deriv_u(cluster, k, kl, i, toggle=1):
            ss = calc_ss(cluster, i)
            uu_perp_deriv_u = calc_uu_perp_deriv_u(cluster, k, kl, i)
            ssuu_perp_deriv_u = ss * uu_perp_deriv_u
            return toggle*ssuu_perp_deriv_u
        
        def calc_uuu_para_deriv_u(cluster, k, kl, i, toggle=1):
            # portion identical to calc_uu_para_deriv_u
            u_para = calc_uu_para(cluster, i)
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            u_para = np.dot(u_diff, self.uvecs[cluster])
            u_para_deriv_u = (del_ik - del_jk) * self.uvecs[cluster][kl]
            uu_para_deriv_u = 2.0 * u_para * u_para_deriv_u
            # product rule trick
            uuu_para_deriv_u = uu_para_deriv_u * u_para + u_para**2 * u_para_deriv_u
            return toggle*uuu_para_deriv_u
        
        def calc_uuu_perp_deriv_u(cluster, k, kl, i, toggle=1):
            # Note: caveat here, not exactly 3rd-order in perp.!
            # portion identical to calc_uu_perp_deriv_u
            del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
            u_diff = self.states_disp[i][cluster[0]] - self.states_disp[i][cluster[1]]
            uu_diff_deriv_u = 2.0 * (del_ik - del_jk) * u_diff[kl]
            uu_para_deriv_u = calc_uu_para_deriv_u(cluster, k, kl, i)
            uu_perp_deriv_u = uu_diff_deriv_u - uu_para_deriv_u
            # product rule trick
            uu_perp = calc_uu_perp(cluster, i)
            u_para = np.dot(u_diff, self.uvecs[cluster])
            u_para_deriv_u = (del_ik - del_jk) * self.uvecs[cluster][kl]
            uuu_perp_deriv_u = uu_perp_deriv_u * u_para + uu_perp * u_para_deriv_u
            return toggle*uuu_perp_deriv_u
        
        def calc_uuuu_para_deriv_u(cluster, k, kl, i, toggle=1):
            uu_para = calc_uu_para(cluster, i)
            uu_para_deriv_u = calc_uu_para_deriv_u(cluster, k, kl, i)
            uuuu_para_deriv_u = 2.0 * (uu_para * uu_para_deriv_u)
            return toggle*uuuu_para_deriv_u
        
        def calc_uuuu_perp_deriv_u(cluster, k, kl, i, toggle=1):
            uu_perp = calc_uu_perp(cluster, i)
            uu_perp_deriv_u = calc_uu_perp_deriv_u(cluster, k, kl, i)
            uuuu_perp_deriv_u = 2.0 * uu_perp * uu_perp_deriv_u
            return toggle*uuuu_perp_deriv_u
        
        def calc_uuuu_paraperp_deriv_u(cluster, k, kl, i, toggle=1):
            uu_para = calc_uu_para(cluster, i)
            uu_para_deriv_u = calc_uu_para_deriv_u(cluster, k, kl, i)
            uu_perp = calc_uu_perp(cluster, i)
            uu_perp_deriv_u = calc_uu_perp_deriv_u(cluster, k, kl, i)
            uuuu_paraperp_deriv_u = uu_para * uu_perp_deriv_u + uu_perp * uu_para_deriv_u
            return toggle*uuuu_paraperp_deriv_u
        
        ####################################
        
        # generate data matrix
        
        datavec = []
        vals, rows, cols = [], [], []

        eqn_counter = 0
        
        if is_magnetic:
            # Input Jij matrix
            if j_matrix_input:
                n = -1
                for cluster in self.unique_pairs_mag:
                    n += 1
                    if cluster in j_matrix_input.keys():
                        jval_in = j_matrix_input[cluster]
                        rows.append(eqn_counter)
                        cols.append(n)
                        vals.append(1.0)
                        datavec.append(jval_in)
                        eqn_counter += 1
        
        # Energies
        for i in range(self.num_config_mag):

            datavec.append(self.energies[i])

            n = -1
            for cluster in self.unique_pairs_mag:
                n += 1
                if is_magnetic:
                    ss = calc_ss(cluster, i)
                    vals.append(ss)
                    rows.append(eqn_counter)
                    cols.append(n)
            for cluster in self.unique_pairs_mag:
                n += 1
                if is_magnetic:
                    ssu_para = calc_ssu_para(cluster, i)
                    vals.append(ssu_para)
                    rows.append(eqn_counter)
                    cols.append(n)
            for cluster in self.unique_triads_ijk:
                n += 1
                if is_magnetic:
                    ssu_ijk = calc_ssu_ijk(cluster, i)
                    vals.append(ssu_ijk)
                    rows.append(eqn_counter)
                    cols.append(n)
            for cluster in self.unique_pairs_mag:
                n += 1
            for cluster in self.unique_pairs:
                n += 1
                uu_para = calc_uu_para(cluster, i)
                vals.append(uu_para)
                rows.append(eqn_counter)
                cols.append(n)
            for cluster in self.unique_pairs:
                n += 1
                uu_perp = calc_uu_perp(cluster, i)
                vals.append(uu_perp)
                rows.append(eqn_counter)
                cols.append(n)
            # magnetic scalar
            n += 1
            if is_magnetic:
                vals.append(1.0)
                rows.append(eqn_counter)
                cols.append(n)
            # nonmagnetic scalar
            n += 1
            if not is_magnetic:
                vals.append(1.0)
                rows.append(eqn_counter)
                cols.append(n)
            #
            eqn_counter += 1

        # Spin derivatives
        if self.hmags:
            for i in range(self.num_config_mag):
                for k in range(self.num_spin):
                    for kl in range(3):

                        datavec.append(self.hmags[i][k][kl])

                        n = -1
                        for cluster in self.unique_pairs_mag:
                            n += 1
                            if k in cluster:
                                ss_deriv_s = calc_ss_deriv_s(cluster, k, kl, i)
                                vals.append(ss_deriv_s)
                                rows.append(eqn_counter)
                                cols.append(n)
                        for cluster in self.unique_pairs_mag:
                            n += 1
                            if k in cluster:
                                ssu_para_deriv_s = calc_ssu_para_deriv_s(cluster, k, kl, i)
                                vals.append(ssu_para_deriv_s)
                                rows.append(eqn_counter)
                                cols.append(n)
                        for cluster in self.unique_pairs:
                            n += 1
                        for cluster in self.unique_pairs:
                            n += 1
                        eqn_counter += 1

        # Displacement derivatives
        for i in range(self.num_config_mag):
            for k in range(self.num_site):
                for kl in range(3):

                    datavec.append(self.forces[i][k][kl])

                    n = -1
                    for cluster in self.unique_pairs_mag:
                        n += 1
                    for cluster in self.unique_pairs_mag:
                        n += 1
                        if is_magnetic:
                            if k in cluster:
                                ssu_para_deriv_u = calc_ssu_para_deriv_u(cluster, k, kl, i)
                                vals.append(ssu_para_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    for cluster in self.unique_triads_ijk:
                        n += 1
                        if is_magnetic:
                            # if k == cluster[2]:
                            ssu_ijk_deriv_u = calc_ssu_ijk_deriv_u(cluster, k, kl, i)
                            vals.append(ssu_ijk_deriv_u)
                            rows.append(eqn_counter)
                            cols.append(n)
                    for cluster in self.unique_pairs_mag:
                        n += 1
                        if fit_anharmonic:
                            if is_magnetic:
                                if k in cluster:                        
                                    ssuu_perp_deriv_u = calc_ssuu_perp_deriv_u(cluster, k, kl, i)
                                    vals.append(ssuu_perp_deriv_u)
                                    rows.append(eqn_counter)
                                    cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if k in cluster:
                            uu_para_deriv_u = calc_uu_para_deriv_u(cluster, k, kl, i)
                            vals.append(uu_para_deriv_u)
                            rows.append(eqn_counter)
                            cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if k in cluster:
                            uu_perp_deriv_u = calc_uu_perp_deriv_u(cluster, k, kl, i)
                            vals.append(uu_perp_deriv_u)
                            rows.append(eqn_counter)
                            cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if fit_anharmonic:
                            if k in cluster:
                                uuu_para_deriv_u = calc_uuu_para_deriv_u(cluster, k, kl, i)
                                vals.append(uuu_para_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if fit_anharmonic:
                            if k in cluster:
                                uuu_perp_deriv_u = calc_uuu_perp_deriv_u(cluster, k, kl, i)
                                vals.append(uuu_perp_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if fit_anharmonic:
                            if k in cluster:
                                uuuu_para_deriv_u = calc_uuuu_para_deriv_u(cluster, k, kl, i)
                                vals.append(uuuu_para_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if fit_anharmonic:
                            if k in cluster:
                                uuuu_perp_deriv_u = calc_uuuu_perp_deriv_u(cluster, k, kl, i)
                                vals.append(uuuu_perp_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    for cluster in self.unique_pairs:
                        n += 1
                        if fit_anharmonic:
                            if k in cluster:
                                uuuu_paraperp_deriv_u = calc_uuuu_paraperp_deriv_u(cluster, k, kl, i)
                                vals.append(uuuu_paraperp_deriv_u)
                                rows.append(eqn_counter)
                                cols.append(n)
                    eqn_counter += 1

        row_len, col_len = eqn_counter, self.num_params
        # print("row len, col len = ", row_len, col_len)

        if is_magnetic:
            self.config_matrix_magnetic = coo_matrix(
                (vals, (rows, cols)), shape=(row_len, col_len), dtype=float)
            self.datavec_magnetic = datavec.copy()
        else:
            self.config_matrix_nonmagnetic = coo_matrix(
                (vals, (rows, cols)), shape=(row_len, col_len), dtype=float)
            self.datavec_nonmagnetic = datavec.copy()

    def get_full_constraint_matrix(self):

        self.get_symm_constraint_matrices()

        rows, cols, vals = [],[],[]

        ##
        row_len = 0
        rows.extend(self.j_ij_const.row + row_len)
        row_len += self.j_ij_const.get_shape()[0]
        #
        rows.extend(self.m_ijk_const.row + row_len)
        row_len += self.m_ijk_const.get_shape()[0]
        #
        rows.extend(self.l_ijk_const.row + row_len)
        row_len += self.l_ijk_const.get_shape()[0]
        # 
        rows.extend(self.m_ijk_const.row + row_len)
        row_len += self.m_ijk_const.get_shape()[0]
        #
        for i in [0,1,2,3,4,5,6]:
            rows.extend(self.k_ij_const.row + row_len)
            row_len += self.k_ij_const.get_shape()[0]

        ##
        col_len = 0
        cols.extend(self.j_ij_const.col + col_len)
        col_len += self.j_ij_const.get_shape()[1]
        #
        cols.extend(self.m_ijk_const.col + col_len)
        col_len += self.m_ijk_const.get_shape()[1]
        #
        cols.extend(self.l_ijk_const.col + col_len)
        col_len += self.l_ijk_const.get_shape()[1]
        #
        cols.extend(self.m_ijk_const.col + col_len)
        col_len += self.m_ijk_const.get_shape()[1]
        #
        for i in [0,1,2,3,4,5,6]:
            cols.extend(self.k_ij_const.col + col_len)
            col_len += self.k_ij_const.get_shape()[1]

        ##
        vals.extend(self.j_ij_const.data)
        vals.extend(self.m_ijk_const.data)
        vals.extend(self.l_ijk_const.data)
        vals.extend(self.m_ijk_const.data)
        for i in [0,1,2,3,4,5,6]:
            vals.extend(self.k_ij_const.data)

        # set to number of parameters (including scalars)
        col_len = self.num_params

        self.a_constr = coo_matrix((vals, (rows, cols)), shape=(row_len, col_len), dtype=float)

    def get_symm_constraint_matrices(self):

        bond_tol = 0.05

        ##
        rows, cols, vals = [],[],[]

        eqn_counter = 0
        for ic, icluster in enumerate(self.unique_pairs_mag):
            for jc, jcluster in enumerate(self.unique_pairs_mag):

                iu = self.vvecs[icluster]
                ju = self.vvecs[jcluster]

                ispecs = [str(self.struct_mag[idx].specie) for idx in icluster[0:2]]
                jspecs = [str(self.struct_mag[idx].specie) for idx in jcluster[0:2]]

                if (bond_tol > np.abs(npla.norm(iu) - npla.norm(ju)) / npla.norm(iu)) \
                        and (set(ispecs) == set(jspecs)):
                    rows.extend(2*[eqn_counter])
                    cols.extend([ic, jc])
                    vals.extend([+1.0, -1.0])
                    eqn_counter += 1

        self.j_ij_const = coo_matrix(
            (vals, (rows, cols)), shape=(eqn_counter, len(self.unique_pairs_mag)), dtype=float)

        ##
        rows, cols, vals = [],[],[]

        eqn_counter = 0
        for ic, icluster in enumerate(self.unique_pairs):
            for jc, jcluster in enumerate(self.unique_pairs):

                iu = self.vvecs[icluster]
                ju = self.vvecs[jcluster]

                ispecs = [str(self.struct[idx].specie) for idx in icluster[0:2]]
                jspecs = [str(self.struct[idx].specie) for idx in jcluster[0:2]]

                if (bond_tol > np.abs(npla.norm(iu) - npla.norm(ju)) / npla.norm(iu)) \
                        and (set(ispecs) == set(jspecs)):
                    rows.extend(2*[eqn_counter])
                    cols.extend([ic, jc])
                    vals.extend([+1.0, -1.0])
                    eqn_counter += 1

        self.k_ij_const = coo_matrix(
            (vals, (rows, cols)), shape=(eqn_counter, len(self.unique_pairs)), dtype=float)

        ##
        rows, cols, vals = [],[],[]

        eqn_counter = 0
        for ic, icluster in enumerate(self.unique_pairs_mag):
            for jc, jcluster in enumerate(self.unique_pairs_mag):

                iu = self.vvecs[icluster]
                ju = self.vvecs[jcluster]

                ispecs = [str(self.struct_mag[idx].specie) for idx in icluster[0:2]]
                jspecs = [str(self.struct_mag[idx].specie) for idx in jcluster[0:2]]

                if (bond_tol > np.abs(npla.norm(iu) - npla.norm(ju)) / npla.norm(iu)) \
                        and (set(ispecs) == set(jspecs)):
                    rows.extend(2*[eqn_counter])
                    cols.extend([ic, jc])
                    vals.extend([+1.0, -1.0])
                    eqn_counter += 1

        self.m_ijk_const = coo_matrix(
            (vals, (rows, cols)), shape=(eqn_counter, len(self.unique_pairs_mag)), dtype=float)
        
        ## 
        rows, cols, vals = [],[],[]
        
        eqn_counter = 0
        for ic, icluster in enumerate(self.unique_triads_ijk):
            for jc, jcluster in enumerate(self.unique_triads_ijk):

                ispecs = [str(self.struct[idx].specie) for idx in icluster[0:3]]
                jspecs = [str(self.struct[idx].specie) for idx in jcluster[0:3]]

                if (set(ispecs) == set(jspecs)):

                    iu_ac = self.vvecs[(icluster[2], icluster[0], icluster[-1][0])]
                    iu_bc = self.vvecs[(icluster[2], icluster[1], icluster[-1][1])]
                    iu_ab = iu_ac - iu_bc

                    ju_ac = self.vvecs[(jcluster[2], jcluster[0], jcluster[-1][0])]
                    ju_bc = self.vvecs[(jcluster[2], jcluster[1], jcluster[-1][1])]
                    ju_ab = ju_ac - ju_bc

                    iu_sum = npla.norm(iu_ac) + npla.norm(iu_bc) + npla.norm(iu_ab)
                    ju_sum = npla.norm(ju_ac) + npla.norm(ju_bc) + npla.norm(ju_ab)
                    diff_metric = np.abs(iu_sum - ju_sum) / iu_sum

                    if (bond_tol > diff_metric):
                        rows.extend(2*[eqn_counter])
                        cols.extend([ic, jc])
                        vals.extend([+1.0, -1.0])
                        eqn_counter += 1

        self.l_ijk_const = coo_matrix(
            (vals, (rows, cols)), shape=(eqn_counter, len(self.unique_triads_ijk)), dtype=float)

class ClusterMatrix:
    def __init__(self, cluster_dict, degree):
        
        self.degree = degree
        self.map = cluster_dict.copy()
        self.vals = []
        self.idxs, self.coords = [], []
        self.idxs_ex, self.coords_ex = [], []
        
        for (idx, coord), val in cluster_dict.items():
            
            # cluster expansion coefficients
            self.vals.append(val)
            
            # site indices
            self.idxs.append(idx)
            self.coords.append(coord)
            
            # site indices - extended
            self.idxs_ex.extend(idx)
            self.coords_ex.extend(coord)
        
        if self.vals:
            self.vals = np.array(self.vals)
            self.idxs_ex, self.coords_ex = np.array(self.idxs_ex, dtype=np.int32), np.array(self.coords_ex, dtype=np.int32)
        else:
            self.vals = np.array([0.0])
            self.idxs_ex, self.coords_ex = np.array(self.degree*[0], dtype=np.int32), np.array(self.degree*[0], dtype=np.int32)

def get_cluster_matrices_iso(j_matrix, jp_matrix, k_para_matrix, k_perp_matrix, uvecs, num_site, num_spin):
    # Generate cluster matrices for
    #     - isotropic Jij exchange
    #     - bond projected model (only displacements along bonds accounted for)
    
    # FIXME: This should be consistent throughout code
    nspindim = 3
    ndispdim = 3
    
    md_ss, md_ssu, md_uu = {}, {}, {}
    
    # spin-spin
    for cluster,v in j_matrix.items():
        i,j = cluster[0],cluster[1]
        for ds in range(nspindim):
            idxs = (i,j)
            coords = (ds,ds)
            md_ss[(idxs, coords)] = v
    
    # spin-spin-displace
    for cluster,v in jp_matrix.items():
        i,j = cluster[0],cluster[1]
        for ds in range(nspindim):
            for du in range(ndispdim):
                #
                idxs = (i,j,i)
                coords = (ds,ds,du)
                md_ssu[(idxs, coords)] = v * uvecs[cluster][du]
                #
                idxs = (i,j,j)
                coords = (ds,ds,du)
                md_ssu[(idxs, coords)] = -v * uvecs[cluster][du]
    
    # displace-displace
    
    # off-diagonal
    for cluster,v in k_para_matrix.items():
        i,j = cluster[0],cluster[1]
        for dua in range(ndispdim):
            for dub in range(ndispdim):
                idxs = (i,j)
                coords = (dua,dub)
                if i != j:
                    value = - 2.0 * v * uvecs[cluster][dua] * uvecs[cluster][dub]
                    md_uu[(idxs, coords)] = value
    # diagonal
    for i in range(num_site):
        for dua in range(ndispdim):
            for dub in range(ndispdim):
                # if dua == dub:  # HACK
                idxs = (i,i)
                coords = (dua,dub)
                value = 0.0
                for cluster,vv in k_para_matrix.items():
                    ii,jj = cluster[0],cluster[1]
                    if ii == i:
                        value += 1.0 * vv * uvecs[cluster][dua] * uvecs[cluster][dub]
                    if jj == i:
                        value += 1.0 * vv * uvecs[cluster][dua] * uvecs[cluster][dub]
                md_uu[(idxs, coords)] = value
    
    # Define cluster matrices
    m_ss = ClusterMatrix(md_ss, degree=2)
    m_ssu = ClusterMatrix(md_ssu, degree=3)
    m_uu = ClusterMatrix(md_uu, degree=2)
    
    return m_ss, m_ssu, m_uu

def get_disp_forces_bond_proj(
    j_matrix, jp_matrix, k_matrix, uvecs,
    state_magn, state_disp):
    
    disp_force_array = np.zeros(np.array(state_disp).shape)
    
    for k in range(len(state_disp)):
        for kl in range(3):
        
            for cluster in jp_matrix.keys():
                s_ip = np.dot(
                    state_magn[cluster[0]], 
                    state_magn[cluster[1]])
                del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
                u_proj_p = (del_ik - del_jk) * uvecs[cluster][kl]
                disp_force_array[k][kl] += jp_matrix[cluster] * s_ip * u_proj_p

            for cluster in k_matrix.keys():
                u_proj = state_disp[cluster[0]] - state_disp[cluster[1]]
                u_proj = np.dot(u_proj, uvecs[cluster])
                del_ik, del_jk = int(cluster[0]==k), int(cluster[1]==k)
                u_proj_p = (del_ik - del_jk) * uvecs[cluster][kl]
                disp_force_array[k][kl] += k_matrix[cluster] * 2.0 * u_proj * u_proj_p
    
    return disp_force_array

def get_energy_bond_proj(
    j_matrix, jp_matrix, k_matrix, uvecs,
    state_magn, state_disp):
    
    e_ss, e_ssu, e_uu = 0.0, 0.0, 0.0
    
    for cluster in j_matrix.keys():
        s_ip = np.dot(
            state_magn[cluster[0]], 
            state_magn[cluster[1]])
        e_ss += j_matrix[cluster] * s_ip
    
    for cluster in jp_matrix.keys():
        s_ip = np.dot(
            state_magn[cluster[0]], 
            state_magn[cluster[1]])
        u_proj = state_disp[cluster[0]] - state_disp[cluster[1]]
        u_proj = np.dot(u_proj, uvecs[cluster])
        s_ip_exp = s_ip * u_proj
        e_ssu += jp_matrix[cluster] * s_ip_exp
    
    for cluster in k_matrix.keys():
        u_proj = state_disp[cluster[0]] - state_disp[cluster[1]]
        u_proj = np.dot(u_proj, uvecs[cluster])
        e_uu += k_matrix[cluster] * u_proj**2
    
    energy = e_ss + e_ssu + e_uu
    
    return (energy, (e_ss, e_ssu, e_uu))

def get_energy_cluster(
    m_ss, m_ssu, m_uu,
    state_magn, state_disp):
    
    e_ss, e_ssu, e_uu = 0.0, 0.0, 0.0
    
    for idx,coord,v in zip(m_ss.idxs, m_ss.coords, m_ss.vals):
        e_ss += v * state_magn[idx[0]][coord[0]] * state_magn[idx[1]][coord[1]]
    
    for idx,coord,v in zip(m_ssu.idxs, m_ssu.coords, m_ssu.vals):
        e_ssu += v * state_magn[idx[0]][coord[0]] * state_magn[idx[1]][coord[1]] * state_disp[idx[2]][coord[2]]
    
    for idx,coord,v in zip(m_uu.idxs, m_uu.coords, m_uu.vals):
        e_uu += v * state_disp[idx[0]][coord[0]] * state_disp[idx[1]][coord[1]]
    
    energy = e_ss + e_ssu + e_uu
    
    return (energy, (e_ss, e_ssu, e_uu))

def get_pairwise_matrix(pairwise_dict, num_site, nx_super, structure):
    
    num_site_super = num_site * np.prod(nx_super)
    
    nx_lim = [nx_super[d] for d in [0,1,2]]
    # nx_lim = [nx_super[d]//2 for d in [0,1,2]]
    
    dist_lim = np.min(
        np.array(nx_super) * np.array(
            [structure.lattice.a, structure.lattice.b, structure.lattice.c]))
    # dist_lim *= 0.5
    
    pw_matrix = {}
    rowcols = []
    # cols, rows, vals = [], [], []
    
    # note: exchange values are already in eV
    for (ii,jj,image),v in pairwise_dict.items():
        
        # any modifications to value here:
        vv = v
        
        # check distance
        svec = np.dot(structure.lattice.matrix.T, image)
        vvec = structure[jj].coords - structure[ii].coords + svec
        # print(1.0e3 * vv, npla.norm(vvec), npla.norm(svec), structure[ii], structure[jj])
        
        if (npla.norm(vvec) <= dist_lim) and all(np.abs(image) <= nx_lim):
            for ni in range(nx_super[0]):
                for nj in range(nx_super[1]):
                    for nk in range(nx_super[2]):
                        nn = [ni, nj, nk]
                        img = [(nn[d]+image[d])%nx_super[d] for d in [0,1,2]]
                        row = get_serial_index(ii,  nn, nx_super, num_site_super)
                        col = get_serial_index(jj, img, nx_super, num_site_super)
                        
                        key = (row,col)
                        
                        pw_matrix[key] = vv
                        
                        # if key in pw_matrix:
                        #     pw_matrix[key] += vv
                        # else:
                        #     pw_matrix[key] = vv
                        
                        # vals.append(vv)
                        # rows.append(row)
                        # cols.append(col)
                        
                        rowcols.append((row,col))
    
    # pw_matrix = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(num_site_super, num_site_super))
    
    # print("index length check:", len(rowcols), len(set(rowcols)))
    
    return pw_matrix

def get_mag_structures(struct_in):
    
    sites_mag, sites_nonmag = [], []
    for site in struct_in:
        if npla.norm(site.properties['magmom']) > 0.0:
            sites_mag.append(site)
        else:
            sites_nonmag.append(site)
    struct = Structure.from_sites(sites=sites_mag+sites_nonmag)
    struct_mag = Structure.from_sites(sites=sites_mag)
    struct_nonmag = Structure.from_sites(sites=sites_nonmag)
    
    return struct, struct_mag, struct_nonmag

def get_serial_index(site_idx, nv, ndims, ns):
    nserial = nv[2] + ndims[2]*(nv[1] + ndims[1]*(nv[0] + ndims[0]*site_idx))
    # nserial = site_idx + ns*(nv[0] + ndims[0]*(nv[1] + ndims[1]*nv[2]))
    return nserial

def get_super_pairwise(pw_dict, struct, nx_super):
    
    num_site = len(struct)
    num_site_super = num_site * np.prod(nx_super)
    
    pw_dict_super = {}
    
    for (i,j,image),v in pw_dict.items():
        for ni in range(nx_super[0]):
            for nj in range(nx_super[1]):
                for nk in range(nx_super[2]):
                    nn = [ni, nj, nk]
                    img = [(nn[d]+image[d])%nx_super[d] for d in [0,1,2]]
                    row = get_serial_index(i,  nn, nx_super, num_site)
                    col = get_serial_index(j, img, nx_super, num_site)
                    
                    image_super = - np.array(img, int) + np.array([nn[d]+image[d] for d in [0,1,2]], int)
                    image_super = [image_super[d]//nx_super[d] for d in [0,1,2]]
                    image_super = tuple(image_super)
                    
                    pw_dict_super[(row, col, image_super)] = v
    
    return pw_dict_super

def get_adj_matrices(struct, nx_super):
    
    num_site = len(struct)
    num_site_super = num_site * np.prod(nx_super)
    
    strategy = MinimumDistanceNN()
    #strategy = MinimumDistanceNN(cutoff=cutoff, get_all_sites=False)
    graph = StructureGraph.with_local_env_strategy(struct, strategy=strategy)
    a_adj_tri = graph.as_dict()['graphs']['adjacency']
    a_adj_tri_super = [[] for i in range(num_site_super)]
    
    rows,cols,vals = [],[],[]
    
    a_adj_super = [[] for i in range(num_site_super)]
    for i in range(num_site):
        for link in a_adj_tri[i]:
            #print(link)
            image = link["to_jimage"]
            j = link['id']
            for ni in range(nx_super[0]):
                for nj in range(nx_super[1]):
                    for nk in range(nx_super[2]):
                        nn = [ni, nj, nk]
                        img = [(nn[d]+image[d])%nx_super[d] for d in [0,1,2]]
                        row = get_serial_index(i,  nn, nx_super, num_site)
                        col = get_serial_index(j, img, nx_super, num_site)
                        a_adj_super[row].append(col)
                        a_adj_super[col].append(row)
                        
                        link_super = {}
                        image_super = - np.array(img, int) + np.array([nn[d]+image[d] for d in [0,1,2]], int)
                        image_super = [image_super[d]//nx_super[d] for d in [0,1,2]]
                        image_super = tuple(image_super)
                        link_super['to_jimage'] = image_super
                        link_super['id'] = col
                        link_super['key'] = link['key']
                        a_adj_tri_super[row].append(link_super.copy())
                        
                        rows.extend([row,col])
                        cols.extend([col,row])
                        vals.extend([1,1])
    a_adj_super = [list(set(a)) for a in a_adj_super]
    a_adj_super_coo = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(num_site_super, num_site_super))
    
    return a_adj_tri, a_adj_tri_super, a_adj_super, a_adj_super_coo


# def pos_to_spins(pos, nspacedim=3):
#     nspins = len(pos) // nspacedim
#     spins = [np.array(pos[i:i+nspacedim].copy()) for i in range(0, len(pos), nspacedim)]
#     return spins

# def struct_from_spins (states, nspacedim=3):

#     nd = nspacedim
#     magmoms = [state[i].s for i in range(num_spin)]

#     struct_final = Structure(species=struct_mag.species, 
#                              coords=struct_mag.frac_coords,
#                              lattice=struct_mag.lattice)
#     struct_final.add_site_property('magmom', magmoms)
#     sites_mag = [site for site in struct_final]

#     sites_nonmag = []
#     for site in struct:
#         for k in magmom_d:
#             if magmom_d[k] == 0.0 and str(site.specie) == k:
#                 sites_nonmag.append(site)

#     sites = sites_mag
#     sites.extend(sites_nonmag)

#     struct = Structure.from_sites(sites)

#     return struct
