"""
"""

import os
import numpy as np

from typing import Dict, Any, Tuple, Sequence, Union
from monty.json import MontyDecoder, MSONable

from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.util.string import str_delimited

import matplotlib.pyplot as plt

__author__ = "Guy C. Moore"
__version__ = "0.0"
__maintainer__ = "Guy C. Moore"
__email__ = "gmoore@lbl.gov"
__status__ = "Development"
__date__ = "March 2021"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class Wannier90win(dict, MSONable):
    """
    A class (heavily based on pymatgen's Incar class) for generating a light-weight
    Wannier90 input file using orbital-like projections of valence shell.
    """

    def __init__(self, params: Dict[str, Any] = None, structure: Structure = None):
        """_summary_

        Args:
            params (Dict[str, Any], optional): _description_. Defaults to None.
            structure (Structure, optional): _description_. Defaults to None.
        """
        super().__init__()

        if params:
            self.update(params)

        if structure:
            self.structure = structure.copy()

    def get_lines_for_projections(self):
        """_summary_

        Orbital-like valence projections

        Returns:
            _type_: _description_
        """

        num_projections = 0
        lines = []
        lines.append("begin projections")

        ang_mom_dict = {"s": 0, "p": 1, "d": 2, "f": 3}

        for site in self.structure:
            f_coords = site.frac_coords.copy()

            l_quant_num = ang_mom_dict[site.specie.block]
            num_ml = 2 * l_quant_num + 1

            for ml_shift in range(num_ml):
                line = "f=%.6f,%.6f,%.6f: l=%i, mr=%i" % (
                    f_coords[0], f_coords[1], f_coords[2],
                    l_quant_num, ml_shift + 1,
                )
                lines.append(line)

                num_projections += 1

        self.num_projections = num_projections
        lines.append("end projections")

        return lines

    def get_string(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """

        lines_proj = self.get_lines_for_projections()
        self["num_wann"] = self.num_projections

        keys = list(self)
        lines_header = []

        for k in keys:
            lines_header.append([k, self[k]])

        str_header = str_delimited(lines_header, None, " = ") + "\n" + "\n"
        str_projections = str_delimited([[l] for l in lines_proj], None, "") + "\n"

        str_full = str_header + str_projections

        return str_full


def get_valence_projected_dos(cdos):
    """_summary_

    Args:
        cdos (_type_): _description_

    Returns:
        _type_: _description_
    """
    ang_mom_dict = {"s": 0, "p": 1, "d": 2, "f": 3}

    dos_outer = {}

    for site in cdos.structure:
        l_quant_num = ang_mom_dict[site.specie.block]
        proj_key = "%s-%s" % (str(site.specie), site.specie.block)

        dos = cdos.get_element_spd_dos(el=site.specie)
        dos_outer[proj_key] = list(dos.items())[l_quant_num][-1]

    return dos_outer


def get_dis_windows(energies, rho_sum, rho_tot, min_frac=0.25, rho_tot_tol=1.0e-2):
    """_summary_

    Args:
        energies (_type_): _description_
        rho_sum (_type_): _description_
        rho_tot (_type_): _description_
        min_frac (float, optional): _description_. Defaults to 0.25.
        rho_tot_tol (float, optional): _description_. Defaults to 1.0e-2.

    Returns:
        _type_: _description_
    """
    rho_rel = np.divide(rho_sum, rho_tot, out=np.zeros_like(rho_tot), where=(rho_tot != 0))

    energy_tol = 2.0 * (energies[1] - energies[0])
    index_mid = np.where(energy_tol > np.abs(energies))[0][0]

    energy_dis_min, energy_dis_max = 0.0, 0.0

    # get lower dis. window
    for i in range(len(energies)):
        index = index_mid - i
        break_out = False
        if index > 0:
            # print(rho_rel[index], rho_tot[index], break_out)
            if (rho_rel[index] > min_frac) or (rho_tot[index] < rho_tot_tol):
                energy_dis_min = energies[index]
            else:
                break_out = True
        if break_out:
            break

    # get upper dis. window
    for i in range(len(energies)):
        index = index_mid + i
        break_out = False
        if index < len(energies):
            # print(rho_rel[index], rho_tot[index], break_out)
            if (rho_rel[index] > min_frac) or (rho_tot[index] < rho_tot_tol):
                energy_dis_max = energies[index]
            else:
                break_out = True
        if break_out:
            break

    # plt.figure();
    # plt.plot(energies, rho_rel);
    # plt.xlim([energy_dis_min, energy_dis_max]);

    return energy_dis_min, energy_dis_max


def analyze_pdos_for_disentangle(vasprun):
    """_summary_

    Args:
        vasprun (_type_): _description_
    """
    smearing = 0.0

    ylim = None
    xlim = None
    # xlim = [-15.0, 15.0]

    cdos = vasprun.complete_dos

    dosplt = DosPlotter(sigma=smearing)
    dos = get_valence_projected_dos(cdos)
    dosplt.add_dos_dict(dos_dict=dos)
    dosplt.add_dos(dos=cdos, label="DOS")

    # dosplt.show(xlim=xlim, ylim=ylim);

    dosplt_dict = dosplt.get_dos_dict().copy()

    energies = np.array(dosplt_dict["DOS"]["energies"])
    density_tot_up = np.array(dosplt_dict["DOS"]["densities"]["1"])
    density_tot_dn = np.array(dosplt_dict["DOS"]["densities"]["-1"])

    density_sum_up, density_sum_dn = np.zeros_like(energies), np.zeros_like(energies)

    for k in dosplt_dict.keys():
        if k != "DOS":
            density_sum_up += np.array(dosplt_dict[k]["densities"]["1"])
            density_sum_dn += np.array(dosplt_dict[k]["densities"]["-1"])

    energy_dis_min_up, energy_dis_max_up = get_dis_windows(energies, density_sum_up, density_tot_up)
    energy_dis_min_dn, energy_dis_max_dn = get_dis_windows(energies, density_sum_dn, density_tot_dn)

    energy_dis_min = max(energy_dis_min_up, energy_dis_min_dn)
    energy_dis_max = min(energy_dis_max_up, energy_dis_max_dn)

    xlim = [energy_dis_min, energy_dis_max]
    # print("energy window:", energy_dis_min, energy_dis_max)

    plt.rcParams["figure.dpi"] = 300

    plt.figure();
    plt.plot(energies, density_sum_up, "b.-");
    plt.plot(energies, -density_sum_dn, "b.-");
    plt.plot(energies, density_tot_up, "r.-");
    plt.plot(energies, -density_tot_dn, "r.-");
    plt.xlim(xlim);
