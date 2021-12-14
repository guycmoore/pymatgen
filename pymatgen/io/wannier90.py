# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Modules for working with wannier90 input and output.
"""

from typing import Sequence

import numpy as np
# from scipy.io import FortranEOFError, FortranFile

import abc
from typing import Dict, Any, Tuple, Sequence, Union

from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import loadfn

from pymatgen.util.typing import PathLike, ArrayLike


__author__ = "Mark Turiansky"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__status__ = "Production"
__date__ = "Jun 04, 2020"


# class Wannier90Input(dict, MSONable):
#     """
#     """

#     def __init__(self, win_file, optional_files=None, **kwargs):
#         """
#         """
#         super().__init__(**kwargs)
#         self.update({"WIN": win_file})
#         if optional_files is not None:
#             self.update(optional_files)

#     def __str__(self):
#         output = []
#         for k, v in self.items():
#             output.append(k)
#             output.append(str(v))
#             output.append("")
#         return "\n".join(output)

#     def as_dict(self):
#         """
#         :return: MSONable dict.
#         """
#         d = {k: v.as_dict() for k, v in self.items()}
#         d["@module"] = self.__class__.__module__
#         d["@class"] = self.__class__.__name__
#         return d

#     @classmethod
#     def from_dict(cls, d):
#         """
#         """
#         dec = MontyDecoder()
#         sub_d = {"optional_files": {}}
#         for k, v in d.items():
#             if k in ["WIN"]:
#                 sub_d[k.lower()] = dec.process_decoded(v)
#             elif k not in ["@module", "@class"]:
#                 sub_d["optional_files"][k] = dec.process_decoded(v)
#         return cls(**sub_d)

#     def write_input(self, output_dir=".", make_dir_if_not_present=True):
#         """
#         """
#         if make_dir_if_not_present and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         for k, v in self.items():
#             if v is not None:
#                 with zopen(os.path.join(output_dir, k), "wt") as f:
#                     f.write(v.__str__())

#     @staticmethod
#     def from_directory(input_dir, optional_files=None):
#         """
#         """
#         sub_d = {}
#         for fname, ftype in [
#             ("WIN", WinFile)
#         ]:
#             try:
#                 fullzpath = zpath(os.path.join(input_dir, fname))
#                 sub_d[fname.lower()] = ftype.from_file(fullzpath)
#             except FileNotFoundError:
#                 sub_d[fname.lower()] = None
#                 pass

#         sub_d["optional_files"] = {}
#         if optional_files is not None:
#             for fname, ftype in optional_files.items():
#                 sub_d["optional_files"][fname] = ftype.from_file(os.path.join(input_dir, fname))
#         return Wannier90Input(**sub_d)

#     def run_wannier90(
#         self,
#         run_dir: PathLike = ".",
#         wannier90_cmd: list = None,
#         output_file: PathLike = "wannier90_run.out",
#         err_file: PathLike = "wannier90_run.err",
#     ):
#         """
#         Write input files and run Wannier90.

#         :param run_dir: Where to write input files and do the run.
#         :param wannier90_cmd: Args to be supplied to run Wannier90. Otherwise, the
#             PMG_WANNIER90_EXE in .pmgrc.yaml is used.
#         :param output_file: File to write output.
#         :param err_file: File to write err.
#         """
#         self.write_input(output_dir=run_dir)
#         wannier90_cmd = wannier90_cmd or SETTINGS.get("PMG_WANNIER90_EXE")
#         wannier90_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in wannier90_cmd]
#         if not wannier90_cmd:
#             raise RuntimeError("You need to supply wannier90_cmd or set the PMG_WANNIER90_EXE in .pmgrc.yaml to run Wannier90.")
#         with cd(run_dir):
#             with open(output_file, "w") as f_std, open(err_file, "w", buffering=1) as f_err:
#                 subprocess.check_call(wannier90_cmd, stdout=f_std, stderr=f_err)


# class Wannier90inputSet(MSONable, metaclass=abc.ABCMeta):
#     """
#     """

#     @property
#     @abc.abstractmethod
#     def win_file(self):
#         """ """
#         pass

#     def get_w90_input(self) -> Wannier90Input:
#         """
#         """
#         return Wannier90Input(
#             win_file=self.win_input_file
#         )

#     def write_input(
#         self,
#         output_dir,
#         make_dir_if_not_present=True,
#         zip_output=False,
#     ):
#         """
#         """

#         vinput = self.get_w90_input()
#         vinput.write_input(output_dir, make_dir_if_not_present=make_dir_if_not_present)

#         if zip_output:
#             filename = self.__class__.__name__ + ".zip"
#             with ZipFile(filename, "w") as zip:
#                 for file in [
#                     "wannier90.win"
#                 ]:
#                     try:
#                         zip.write(file)
#                         os.remove(file)
#                     except FileNotFoundError:
#                         pass

#     def as_dict(self, verbosity=2):
#         """
#         """
#         d = MSONable.as_dict(self)
#         if verbosity == 1:
#             d.pop("structure", None)
#         return d


# class WinFile(dict, MSONable):
#     """
#     Wannier90 input file object for reading and writing wannier90.win files. Essentially consists of
#     a dictionary with some helper functions
#     """

#     def __init__(self, params: Dict[str, Any] = None):
#         """
#         Creates an WinFile object.

#         Args:
#             params (dict): A set of input parameters as a dictionary.
#         """
#         super().__init__()
#         if params:

#             self.update(params)

#     def __setitem__(self, key: str, val: Any):
#         """
#         Add parameter-val pair to WinFile.  Warns if parameter is not in list of
#         valid wannier90.win tags. Also cleans the parameter and val by stripping
#         leading and trailing white spaces.
#         """
#         super().__setitem__(
#             key.strip(),
#             WinFile.proc_val(key.strip(), val.strip()) if isinstance(val, str) else val,
#         )

#     def as_dict(self) -> dict:
#         """
#         :return: MSONable dict.
#         """
#         d = dict(self)
#         d["@module"] = self.__class__.__module__
#         d["@class"] = self.__class__.__name__
#         return d

#     @classmethod
#     def from_dict(cls, d) -> "WinFile":
#         """
#         :param d: Dict representation.
#         :return: WinFile
#         """

#         return WinFile({k: v for k, v in d.items() if k not in ("@module", "@class")})

#     def get_string(self, sort_keys: bool = False, pretty: bool = False) -> str:
#         """
#         Returns a string representation of the wannier90.win.  The reason why this
#         method is different from the __str__ method is to provide options for
#         pretty printing.

#         Args:
#             sort_keys (bool): Set to True to sort the wannier90.win parameters
#                 alphabetically. Defaults to False.
#             pretty (bool): Set to True for pretty aligned output. Defaults
#                 to False.
#         """
#         keys = list(self.keys())
#         if sort_keys:
#             keys = sorted(keys)
#         lines = []
#         for k in keys:
#             lines.append([k, self[k]])

#         if pretty:
#             return str(tabulate([[l[0], "=", l[1]] for l in lines], tablefmt="plain"))
#         return str_delimited(lines, None, " = ") + "\n"

#     def __str__(self):
#         return self.get_string(sort_keys=True, pretty=False)

#     def write_file(self, filename: PathLike):
#         """
#         Write WinFile to a wannier90.win file.

#         Args:
#             filename (str): filename to write to.
#         """
#         with zopen(filename, "wt") as f:
#             f.write(self.__str__())

#     @staticmethod
#     def from_file(filename: PathLike) -> "WinFile":
#         """
#         Reads an WinFile object from a wannier90.win file.

#         Args:
#             filename (str): Filename for file

#         Returns:
#             WinFile object
#         """
#         with zopen(filename, "rt") as f:
#             return WinFile.from_string(f.read())

#     @staticmethod
#     def from_string(string: str) -> "WinFile":
#         """
#         Reads an WinFile object from a string.

#         Args:
#             string (str): WinFile string

#         Returns:
#             WinFile object
#         """
#         lines = list(clean_lines(string.splitlines()))
#         params = {}
#         for line in lines:
#             for sline in line.split(";"):
#                 m = re.match(r"(\w+)\s*=\s*(.*)", sline.strip())
#                 if m:
#                     key = m.group(1).strip()
#                     val = m.group(2).strip()
#                     val = WinFile.proc_val(key, val)
#                     params[key] = val
#         return WinFile(params)

#     @staticmethod
#     def proc_val(key: str, val: Any):
#         """
#         Static helper method to convert wannier90.win parameters to proper types, e.g.,
#         integers, floats, lists, etc.

#         Args:
#             key: wannier90.win parameter key
#             val: Actual value of wannier90.win parameter.
#         """

#         # Note: add 'restart'
#         # list_keys = (
#         #     "LDAUU",
#         # )
#         bool_keys = (
#             "guiding_centres",
#             "write_xyz",
#             "write_hr",
#         )
#         float_keys = (
#             "dis_win_min",
#             "dis_win_max",
#             "kmesh_tol",
#         )
#         int_keys = (
#             "num_iter",
#             "num_print_cycles",
#             "num_bands",
#             "num_wann",
#         )

#         def smart_int_or_float(numstr):
#             if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
#                 return float(numstr)
#             return int(numstr)

#         # try:
#         #     if key in list_keys:
#         #         output = []
#         #         toks = re.findall(r"(-?\d+\.?\d*)\*?(-?\d+\.?\d*)?\*?(-?\d+\.?\d*)?", val)
#         #         for tok in toks:
#         #             if tok[2] and "3" in tok[0]:
#         #                 output.extend([smart_int_or_float(tok[2])] * int(tok[0]) * int(tok[1]))
#         #             elif tok[1]:
#         #                 output.extend([smart_int_or_float(tok[1])] * int(tok[0]))
#         #             else:
#         #                 output.append(smart_int_or_float(tok[0]))
#         #         return output
#         #     if key in bool_keys:
#         #         m = re.match(r"^\.?([T|F|t|f])[A-Za-z]*\.?", val)
#         #         if m:
#         #             return m.group(1).lower() == "t"

#         #         raise ValueError(key + " should be a boolean type!")

#         #     if key in float_keys:
#         #         return float(re.search(r"^-?\d*\.?\d*[e|E]?-?\d*", val).group(0))  # type: ignore

#         #     if key in int_keys:
#         #         return int(re.match(r"^-?[0-9]+", val).group(0))  # type: ignore

#         # except ValueError:
#         #     pass

#         # Not in standard keys. We will try a hierarchy of conversions.
#         try:
#             val = int(val)
#             return val
#         except ValueError:
#             pass

#         try:
#             val = float(val)
#             return val
#         except ValueError:
#             pass

#         if "true" in val.lower():
#             return True

#         if "false" in val.lower():
#             return False

#         return val.strip().capitalize()

#     def diff(self, other: "WinFile") -> Dict[str, Dict[str, Any]]:
#         """
#         Diff function for WinFile.  Compares two WinFiles and indicates which
#         parameters are the same and which are not. Useful for checking whether
#         two runs were done using the same parameters.

#         Args:
#             other (WinFile): The other WinFile object to compare to.

#         Returns:
#             Dict of the following format:
#             {"Same" : parameters_that_are_the_same,
#             "Different": parameters_that_are_different}
#             Note that the parameters are return as full dictionaries of values.
#         """
#         similar_param = {}
#         different_param = {}
#         for k1, v1 in self.items():
#             if k1 not in other:
#                 different_param[k1] = {"WINFILE1": v1, "WINFILE2": None}
#             elif v1 != other[k1]:
#                 different_param[k1] = {"WINFILE1": v1, "WINFILE2": other[k1]}
#             else:
#                 similar_param[k1] = v1
#         for k2, v2 in other.items():
#             if k2 not in similar_param and k2 not in different_param:
#                 if k2 not in self:
#                     different_param[k2] = {"WINFILE1": None, "WINFILE2": v2}
#         return {"Same": similar_param, "Different": different_param}

#     def __add__(self, other):
#         """
#         Add all the values of another WinFile object to this object.
#         Facilitates the use of "standard" WinFiles.
#         """
#         params = dict(self.items())
#         for k, v in other.items():
#             if k in self and v != self[k]:
#                 raise ValueError("WinFiles have conflicting values!")
#             params[k] = v
#         return WinFile(params)

#     def check_params(self):
#         """
#         Raises a warning for nonsensical or non-existant wannier90.win tags and
#         parameters.
#         """
#         for k in self.keys():

#             # First check if this parameter even exists
#             if k not in incar_params.keys():
#                 warnings.warn(
#                     "Cannot find %s in the list of wannier90.win flags" % (k),
#                     BadWinFileWarning,
#                     stacklevel=2,
#                 )

#             if k in incar_params.keys():
#                 if type(incar_params[k]).__name__ == "str":
#                     # Now we check if this is an appropriate parameter type
#                     if incar_params[k] == "float":
#                         if not type(self[k]) not in ["float", "int"]:
#                             warnings.warn(
#                                 "%s: %s is not real" % (k, self[k]),
#                                 BadWinFileWarning,
#                                 stacklevel=2,
#                             )
#                     elif type(self[k]).__name__ != incar_params[k]:
#                         warnings.warn(
#                             "%s: %s is not a %s" % (k, self[k], incar_params[k]),
#                             BadWinFileWarning,
#                             stacklevel=2,
#                         )

#                 # if we have a list of possible parameters, check
#                 # if the user given parameter is in this list
#                 elif type(incar_params[k]).__name__ == "list":
#                     if self[k] not in incar_params[k]:
#                         warnings.warn(
#                             "%s: Cannot find %s in the list of parameters" % (k, self[k]),
#                             BadWinFileWarning,
#                             stacklevel=2,
#                         )

class Unk:
    """
    Object representing the data in a UNK file.

    .. attribute:: ik

        int index of kpoint for this file

    .. attribute:: data

        numpy.ndarray that contains the wavefunction data for in the UNK file.
        The shape should be (nbnd, ngx, ngy, ngz) for regular calculations and
        (nbnd, 2, ngx, ngy, ngz) for noncollinear calculations.

    .. attribute:: is_noncollinear

        bool that specifies if data is from a noncollinear calculation

    .. attribute:: nbnd

        int number of bands in data

    .. attribute:: ng

        sequence of three integers that correspond to the grid size of the
        given data. The definition is ng = (ngx, ngy, ngz).

    """

    ik: int
    is_noncollinear: bool
    nbnd: int
    ng: Sequence[int]

    def __init__(self, ik: int, data: np.ndarray) -> None:
        """
        Initialize Unk class.

        Args:
            ik (int): index of the kpoint UNK file is for
            data (np.ndarray): data from the UNK file that has shape (nbnd,
                ngx, ngy, ngz) or (nbnd, 2, ngx, ngy, ngz) if noncollinear
        """
        self.ik = ik
        self.data = data

    @property
    def data(self) -> np.ndarray:
        """
        np.ndarray: contains the wavefunction data for in the UNK file.
        The shape should be (nbnd, ngx, ngy, ngz) for regular calculations and
        (nbnd, 2, ngx, ngy, ngz) for noncollinear calculations.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """
        Sets the value of data.

        Args:
            value (np.ndarray): data to replace stored data, must haveshape
                (nbnd, ngx, ngy, ngz) or (nbnd, 2, ngx, ngy, ngz) if
                noncollinear calculation
        """
        temp_val = np.array(value, dtype=np.complex128)
        if len(temp_val.shape) not in [4, 5]:
            raise ValueError(
                "invalid data shape, must be (nbnd, ngx, ngy, ngz"
                ") or (nbnd, 2, ngx, ngy, ngz) for noncollinear "
                f"data, given {temp_val.shape}"
            )
        if len(temp_val.shape) == 5 and temp_val.shape[1] != 2:
            raise ValueError(
                "invalid noncollinear data, shape should be (nbnd" f", 2, ngx, ngy, ngz), given {temp_val.shape}"
            )
        self._data = temp_val

        # derived properties
        self.is_noncollinear = len(self.data.shape) == 5
        self.nbnd = self.data.shape[0]
        self.ng = self.data.shape[-3:]

    @staticmethod
    def from_file(filename: str) -> object:
        """
        Reads the UNK data from file.

        Args:
            filename (str): path to UNK file to read

        Returns:
            Unk object
        """
        input_data = []
        with FortranFile(filename, "r") as f:
            *ng, ik, nbnd = f.read_ints()
            for _ in range(nbnd):
                input_data.append(
                    # when reshaping need to specify ordering as fortran
                    f.read_record(np.complex128).reshape(ng, order="F")
                )
            try:
                for _ in range(nbnd):
                    input_data.append(f.read_record(np.complex128).reshape(ng, order="F"))
                is_noncollinear = True
            except FortranEOFError:
                is_noncollinear = False

        # mypy made me create an extra variable here >:(
        data = np.array(input_data, dtype=np.complex128)

        # spinors are interwoven, need to separate them
        if is_noncollinear:
            temp_data = np.empty((nbnd, 2, *ng), dtype=np.complex128)
            temp_data[:, 0, :, :, :] = data[::2, :, :, :]
            temp_data[:, 1, :, :, :] = data[1::2, :, :, :]
            return Unk(ik, temp_data)
        return Unk(ik, data)

    def write_file(self, filename: str) -> None:
        """
        Write the UNK file.

        Args:
            filename (str): path to UNK file to write, the name should have the
                form 'UNKXXXXX.YY' where XXXXX is the kpoint index (Unk.ik) and
                YY is 1 or 2 for the spin index or NC if noncollinear
        """
        with FortranFile(filename, "w") as f:
            f.write_record(np.array([*self.ng, self.ik, self.nbnd], dtype=np.int32))
            for ib in range(self.nbnd):
                if self.is_noncollinear:
                    f.write_record(self.data[ib, 0].flatten("F"))
                    f.write_record(self.data[ib, 1].flatten("F"))
                else:
                    f.write_record(self.data[ib].flatten("F"))

    def __repr__(self) -> str:
        return (
            f"<UNK ik={self.ik} nbnd={self.nbnd} ncl={self.is_noncollinear}"
            + f" ngx={self.ng[0]} ngy={self.ng[1]} ngz={self.ng[2]}>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unk):
            return NotImplemented

        if not np.allclose(self.ng, other.ng):
            return False

        if self.ik != other.ik:
            return False

        if self.is_noncollinear != other.is_noncollinear:
            return False

        if self.nbnd != other.nbnd:
            return False

        for ib in range(self.nbnd):
            if self.is_noncollinear:
                if not (
                    np.allclose(self.data[ib, 0], other.data[ib, 0], atol=1e-4)
                    and np.allclose(self.data[ib, 1], other.data[ib, 1], atol=1e-4)
                ):
                    return False
            else:
                if not np.allclose(self.data[ib], other.data[ib], atol=1e-4):
                    return False
        return True
