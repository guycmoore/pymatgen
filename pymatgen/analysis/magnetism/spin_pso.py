"""
This module contains all of the necessary functionality for the particle swarm optimization (PSO),
specifically SpinPSO. A lightweight Heisenberg model is included as a potential energy landscape for testing.
"""

import os
import numpy as np

from pymatgen.core import Structure

__author__ = "Guy C. Moore"
__version__ = "0.0"
__maintainer__ = "Guy C. Moore"
__email__ = "gmoore@lbl.gov"
__status__ = "Development"
__date__ = "March 2021"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class OptimizerPSO:
    """
    Class for performing particle swarm optimization (both conventional cartesian PSO, as well as SpinPSO)
    """

    def __init__(self, pot_energy_surface, num_agent, positions_init, pos_dicts, vels_init, nspindim=3):
        """
        Potential energy surface and the swarm are initialized.

        Args:
            pot_energy_surface (_type_): Potential energy surface to minimize over.
            num_agent (int): Number of agents in the swarm.
            positions_init (list): Initial positions of agents on potential energy surface.
            pos_dicts (list): Position descriptor dictionaries (as list), 
                e.g. dimensionality & type ("cartesian", "spin", ...) of the individual 
                positions that make up the global position.
            vels_init (list): Initial velocities of agents on potential energy surface.
            nspindim (int, optional): Dimensionality of spins. Defaults to 3.
        """
        self.pes = pot_energy_surface
        self.swarm = Swarm(num_agent, positions_init, pos_dicts, vels_init, nspindim)

    def optimize(self, fit_history, pos_history, save_pos=False, savefreq=1, max_iter=100, dt=1.0, mass=1.0):
        """_summary_

        Args:
            fit_history (list): List of the swarm's previous "best" fitnesses (lowest energies).
            pos_history (list): List of the previous positions of all agents in the swarm.
            save_pos (bool, optional): _description_. Defaults to False.
            savefreq (int, optional): _description_. Defaults to 1.
            max_iter (int, optional): _description_. Defaults to 100.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
        """
        for n in range(max_iter):
            self.swarm.update_fitnesses(self.pes)
            self.swarm.compute_velocities(gamma=0.5, lam=1.0, dt=dt, mass=mass)
            self.swarm.update_positions(dt)
            if save_pos and np.mod(n, savefreq) == 0:
                fit_history.append(self.swarm.fit_best[0])
                pos_history.append(self.swarm.get_positions())

    def optimize_gcpso(
        self, fit_history, pos_history, grad_history, save_pos=False, savefreq=1, max_iter=100, dt=1.0, mass=1.0
    ):
        """_summary_

        Args:
            fit_history (_type_): _description_
            pos_history (_type_): _description_
            grad_history (_type_): _description_
            save_pos (bool, optional): _description_. Defaults to False.
            savefreq (int, optional): _description_. Defaults to 1.
            max_iter (int, optional): _description_. Defaults to 100.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
        """
        for n in range(max_iter):
            # ########################################
            # # serialization test
            # swarm_d = self.swarm.as_dict()
            # self.swarm = Swarm.from_dict(swarm_d)
            # ########################################

            pos_current = self.swarm.get_positions()
            fits_new = [self.pes.evaluate_energy(p) for p in pos_current]
            grads = [self.pes.compute_gradient(p) for p in pos_current]

            self.swarm.update_fitnesses_gcpso(fits_new)
            self.swarm.compute_velocities_gcpso(grads, gamma=0.5, lam=1.0, dt=dt, mass=mass)
            self.swarm.update_positions(dt)
            if save_pos and np.mod(n, savefreq) == 0:
                fit_history.append(self.swarm.fit_best[0])
                pos_history.append(self.swarm.get_positions())
                grad_history.append(self.pes.compute_gradient(self.swarm.get_pos_best()))


class Swarm:
    def __init__(
        self,
        num_agent, positions_init=None, pos_dicts=None, vels_init=None, nspindim=3,
        agents_in=None, fit_best=None, pos_best=None, index_best=-1,
        rho=1.0, rho_lim=16.0, rho_scale=0.5,
        num_failure=0, num_success=0, failure_lim=3, success_lim=3,
    ):
        """_summary_

        Args:
            num_agent (_type_): _description_
            positions_init (_type_, optional): _description_. Defaults to None.
            pos_dicts (_type_, optional): _description_. Defaults to None.
            vels_init (_type_, optional): _description_. Defaults to None.
            nspindim (int, optional): _description_. Defaults to 3.
            agents_in (_type_, optional): _description_. Defaults to None.
            fit_best (_type_, optional): _description_. Defaults to None.
            pos_best (_type_, optional): _description_. Defaults to None.
            index_best (int, optional): _description_. Defaults to -1.
            rho (float, optional): _description_. Defaults to 1.0.
            rho_lim (float, optional): _description_. Defaults to 16.0.
            rho_scale (float, optional): _description_. Defaults to 0.5.
            num_failure (int, optional): _description_. Defaults to 0.
            num_success (int, optional): _description_. Defaults to 0.
            failure_lim (int, optional): _description_. Defaults to 3.
            success_lim (int, optional): _description_. Defaults to 3.
        """
        if not positions_init:
            positions_init = []
            for agent in agents_in:
                positions_init.append(agent.get_position())

        self.num_agent = num_agent
        self.nquant = len(positions_init[0])
        self.nspindim = nspindim

        self.fit_best = fit_best if fit_best else [float("inf")]
        self.pos_best = pos_best if pos_best else [[float("inf")] for v in positions_init[0]]

        self.index_best = index_best
        self.rho = rho
        self.rho_lim = rho_lim
        self.rho_scale = rho_scale
        self.num_failure = num_failure
        self.num_success = num_success
        self.failure_lim = failure_lim
        self.success_lim = success_lim

        if agents_in:
            self.agents = [agent for agent in agents_in]
        else:
            self.agents = []
            for i in range(num_agent):
                self.agents.append(
                    Agent(
                        pos_init=positions_init[i], pos_dicts=pos_dicts, vel_init=vels_init[i], nspindim=self.nspindim
                    )
                )

    def update_fitnesses(self, fits_new):
        """_summary_

        Args:
            fits_new (_type_): _description_
        """
        for i in range(self.num_agent):
            self.agents[i].update_fitness(fits_new[i], self.fit_best, self.pos_best)

        for i in range(self.num_agent):
            if self.agents[i].get_fitness() <= self.fit_best[0] or self.fit_best[0] == float("inf"):
                self.fit_best[0] = self.agents[i].get_fitness()
                self.pos_best = self.agents[i].get_position()

    def update_fitnesses_gcpso(self, fits_new, pos_new=None):
        """_summary_

        Args:
            fits_new (_type_): _description_
            pos_new (_type_, optional): _description_. Defaults to None.
        """
        fit_best_old = [float("inf")]
        fit_best_old[0] = self.fit_best[0]

        if pos_new:
            for i in range(self.num_agent):
                self.agents[i].overwrite_position(pos_update=pos_new[i])

        for i in range(self.num_agent):
            self.agents[i].update_fitness(fits_new[i], self.fit_best, self.pos_best)

        for i in range(self.num_agent):
            if self.agents[i].get_fitness() < self.fit_best[0] or self.fit_best[0] == float("inf"):
                self.index_best = i
                self.fit_best[0] = self.agents[i].get_fitness()
                self.pos_best = self.agents[i].get_position()

        if self.fit_best[0] == fit_best_old[0]:
            self.num_failure += 1
            self.num_success = 0
        else:
            self.num_failure = 0
            self.num_success += 1

        if self.num_failure > self.failure_lim:
            self.rho *= self.rho_scale

        if self.num_success > self.success_lim and self.rho < self.rho_lim:
            self.rho *= 1.0 / self.rho_scale
            # self.rho *= 1.0

    #         print("fitness, fitness old = ", self.fit_best[0], fit_best_old[0])
    #         print("best index = ", self.index_best)
    #         print("rho = ", self.rho)
    #         print()

    def compute_velocities(self, gamma=0.5, lam=1.0, dt=1.0, mass=1.0):
        """_summary_

        Args:
            gamma (float, optional): _description_. Defaults to 0.5.
            lam (float, optional): _description_. Defaults to 1.0.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
        """
        for i in range(num_agent):
            self.agents[i].compute_vel(self.pos_best, is_best=False, gamma=gamma, lam=lam, dt=dt, mass=mass)

    def compute_velocities_gcpso(self, grads_new, gamma=0.5, lam=1.0, dt=1.0, mass=1.0):
        """_summary_

        Args:
            grads_new (_type_): _description_
            gamma (float, optional): _description_. Defaults to 0.5.
            lam (float, optional): _description_. Defaults to 1.0.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
        """
        for i in range(self.num_agent):
            if i == self.index_best:
                # FIXME: "reset" position to global (& historical) best position
                gradient = grads_new[i]
                self.agents[i].compute_force(self.pos_best, mass=mass)
                self.agents[i].compute_forcebest(
                    self.pos_best,
                    gradient,
                    rho=self.rho,
                    mass=mass,
                )
                self.agents[i].compute_vel(self.pos_best, is_best=True, gamma=gamma, lam=lam, dt=dt, mass=mass)
            else:
                self.agents[i].compute_force(self.pos_best, mass=mass)
                self.agents[i].compute_vel(self.pos_best, is_best=False, gamma=gamma, lam=lam, dt=dt, mass=mass)

    def update_positions(self, dt=1.0):
        """_summary_

        Args:
            dt (float, optional): _description_. Defaults to 1.0.
        """
        for i in range(self.num_agent):
            self.agents[i].update_position(dt=dt)

    def get_positions(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [self.agents[i].get_position() for i in range(self.num_agent)]

    def get_pos_best(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.pos_best

    def as_dict(self):
        """_summary_

        Returns:
            dict: _description_
        """
        swarm_dict = {}

        swarm_dict.update(
            {"dims": {"num_agent": self.num_agent, "num_quant": self.nquant, "num_spindim": self.nspindim}}
        )

        swarm_dict.update({"fit": {"best_fitness": self.fit_best, "best_position": dyn_tolist(self.pos_best)}})

        swarm_dict.update(
            {
                "gcpso": {
                    "best_index": self.index_best,
                    "rho": self.rho,
                    "rho_iteration_scale": self.rho_scale,
                    "rho_limit": self.rho_lim,
                    "num_failure": self.num_failure,
                    "num_success": self.num_success,
                    "failure_limit": self.failure_lim,
                    "success_limit": self.success_lim,
                }
            }
        )

        agent_dicts = []
        for i in range(self.num_agent):
            agent_dicts.append(self.agents[i].as_dict())

        swarm_dict.update({"agents": agent_dicts})

        return swarm_dict

    @classmethod
    def from_dict(Swarm, d):
        """_summary_

        Args:
            Swarm (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        agents = []
        for i in range(len(d["agents"])):
            agents.append(Agent.from_dict(d["agents"][i]))

        return Swarm(
            num_agent=d["dims"]["num_agent"],
            nspindim=d["dims"]["num_spindim"],
            agents_in=agents,
            fit_best=d["fit"]["best_fitness"],
            pos_best=dyn_toarray(d["fit"]["best_position"]),
            index_best=d["gcpso"]["best_index"],
            rho=d["gcpso"]["rho"],
            rho_lim=d["gcpso"]["rho_limit"],
            rho_scale=d["gcpso"]["rho_iteration_scale"],
            num_failure=d["gcpso"]["num_failure"],
            num_success=d["gcpso"]["num_success"],
            failure_lim=d["gcpso"]["failure_limit"],
            success_lim=d["gcpso"]["success_limit"],
        )


class Agent:
    def __init__(
        self, pos_init, pos_dicts, vel_init, force_init=None, nspindim=3,
        w_inertial=0.5, w_cognitive=1.0, w_social=2.0,
        fit=float("inf"), fit_best=float("inf"), pos_best=None,
    ):
        """_summary_

        Args:
            pos_init (_type_): _description_
            pos_dicts (_type_): _description_
            vel_init (_type_): _description_
            force_init (_type_, optional): _description_. Defaults to None.
            nspindim (int, optional): _description_. Defaults to 3.
            w_inertial (float, optional): _description_. Defaults to 0.5.
            w_cognitive (float, optional): _description_. Defaults to 1.0.
            w_social (float, optional): _description_. Defaults to 2.0.
            fit (_type_, optional): _description_. Defaults to float("inf").
            fit_best (_type_, optional): _description_. Defaults to float("inf").
            pos_best (_type_, optional): _description_. Defaults to None.
        """

        self.nquant = len(pos_init)
        self.nspindim = nspindim

        # Position, Velocity, & Force
        self.pos = dyn_copy(pos_init)
        self.vel = dyn_copy(vel_init)
        self.force = dyn_copy(force_init) if force_init else [0.0 * a for a in pos_init]

        self.pos_dicts = pos_dicts

        # Fitness (current) and Best fitness & position
        self.fit = fit if fit else float("inf")
        if fit_best and pos_best:
            self.fit_best = fit_best
            self.pos_best = dyn_copy(pos_best)
        else:
            self.fit_best = float("inf")
            self.pos_best = [[float("inf")]]

        # PSO parameters
        self.w_inertial = w_inertial
        self.w_cognitive = w_cognitive
        self.w_social = w_social

    def overwrite_position(self, pos_update):
        """_summary_

        Args:
            pos_update (_type_): _description_
        """
        self.pos = dyn_copy(pos_update)

    def update_fitness(self, fit_new, fit_best_g, pos_best_g):
        """_summary_

        Args:
            fit_new (_type_): _description_
            fit_best_g (_type_): _description_
            pos_best_g (_type_): _description_
        """
        self.fit = fit_new

        # Update personal best
        if (self.fit < self.fit_best) or self.fit_best == float("inf"):
            self.fit_best = self.fit
            self.pos_best = dyn_copy(self.pos)

        # FIXME: remove here if performed later on (i.e. in GCPSO)
        # # Update global best
        # if (self.fit_best < fit_best_g[0]) or fit_best_g[0] == float('inf'):
        #     fit_best_g[0] = self.fit_best
        #     pos_best_g = dyn_copy(self.pos_best)

    def compute_force(self, pos_best_g, mass=1.0):
        """_summary_

        Args:
            pos_best_g (_type_): _description_
            mass (float, optional): _description_. Defaults to 1.0.

        Raises:
            ValueError: _description_
        """
        for i in range(self.nquant):
            if self.pos_dicts[i]["type"] == "cartesian":
                # Inertial term
                self.force[i] = (self.w_inertial - 1.0) * self.vel[i]

                # Cognition term
                sigma_c = np.array([random.uniform(0.0, 1.0) for i in range(len(self.pos[i]))])
                self.force[i] += self.w_cognitive * sigma_c * (self.pos_best[i] - self.pos[i])

                # Social term
                sigma_s = np.array([random.uniform(0.0, 1.0) for i in range(len(self.pos[i]))])
                self.force[i] += self.w_social * sigma_s * (pos_best_g[i] - self.pos[i])

                # Weight accordingly
                self.force[i] *= mass

            elif self.pos_dicts[i]["type"] == "spin":
                # # Inertial term
                # self.force[i] = self.w_inertial * self.force[i]

                # NOTE: No inertial term in SpinPSO
                self.force[i][0:] = 0.0

                # Cognition term
                sigma_c = 1.0 - random.uniform(0.0, 1.0)  # unif. on (0,1]
                self.force[i] += self.w_cognitive * sigma_c * self.pos_best[i] / npla.norm(self.pos_best[i])

                # Social term
                sigma_s = 1.0 - random.uniform(0.0, 1.0)  # unif. on (0,1]
                self.force[i] += self.w_social * sigma_s * pos_best_g[i] / npla.norm(pos_best_g[i])

                # Weight accordingly
                # self.force[i] *= mass
                self.force[i] /= npla.norm(self.force[i])

            else:
                raise ValueError("Invalid PSO dynamic type.")


    def compute_forcebest(self, pos_best_g, grad, rho=1.0, mass=1.0, gcpso_type="grad"):
        """_summary_

        Note: this function should be performed AFTER compute_force() - adds to previous calculated force

        Args:
            pos_best_g (_type_): _description_
            grad (_type_): _description_
            rho (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
            gcpso_type (str, optional): _description_. Defaults to "grad".

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        for i in range(self.nquant):
            if self.pos_dicts[i]["type"] == "spin":
                if gcpso_type == "grad":
                    f = -8.0 * rho * grad[i] / npla.norm(self.pos[i])
                    # f = - rho * grad[i] / npla.norm(grad[i])
                elif gcpso_type == "random":
                    f = -np.array([random.gauss(0.0, 1.0) for i in range(self.nspindim)])
                    f /= npla.norm(f)
                    f = -rho * f
                else:
                    raise ValueError("Invalid GCPSO type.")

                self.force[i] += f
                self.force[i] /= npla.norm(self.force[i])
            else:
                raise ValueError("Invalid PSO dynamic type.")

    def compute_vel(self, pos_best_g, is_best=False, dt=1.0, mass=1.0, gamma=0.5, lam=1.0):
        """_summary_

        Args:
            pos_best_g (_type_): _description_
            is_best (bool, optional): _description_. Defaults to False.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
            gamma (float, optional): _description_. Defaults to 0.5.
            lam (float, optional): _description_. Defaults to 1.0.

        Raises:
            ValueError: _description_
        """
        for i in range(self.nquant):
            if self.pos_dicts[i]["type"] == "cartesian":
                self.vel[i] = 1.0 * self.vel[i] + dt * self.force[i]

            elif self.pos_dicts[i]["type"] == "spin":
                pos_norm = npla.norm(self.pos[i])
                pos_normed_i = self.pos[i] / pos_norm
                self.vel[i] = -gamma * np.cross(pos_normed_i, self.force[i])
                self.vel[i] += -lam * np.cross(pos_normed_i, np.cross(pos_normed_i, self.force[i]))

                # Correct velocity
                if is_best:
                    pos_predict = pos_best_g[i] / npla.norm(pos_best_g[i]) + dt * self.vel[i]
                else:
                    pos_predict = pos_normed_i + dt * self.vel[i]
                pos_correct = pos_predict.copy()
                pos_correct *= npla.norm(self.pos[i]) / npla.norm(pos_predict)
                self.vel[i] = (pos_correct - self.pos[i]) / dt

            else:
                raise ValueError("Invalid SpinPSO dynamic type.")

    def update_position(self, dt=1.0):
        """_summary_

        Args:
            dt (float, optional): _description_. Defaults to 1.0.
        """
        for i in range(self.nquant):
            self.pos[i] = self.pos[i] + dt * self.vel[i]

    def get_position(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return dyn_copy(self.pos)

    def get_fitness(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.fit

    def as_dict(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        agent_dict = {}

        agent_dict.update({"pes": {"num_quant": self.nquant, "num_spindim": self.nspindim}})

        # Position, Velocity, & Force
        agent_dict.update(
            {
                "dyn": {
                    "position": dyn_tolist(self.pos),
                    "pos_dicts": self.pos_dicts,
                    "velocity": dyn_tolist(self.vel),
                    "force": dyn_tolist(self.force),
                }
            }
        )

        # Fitness (current) and Best fitness & position
        agent_dict.update(
            {"fit": {"fitness": self.fit, "best_fitness": self.fit_best, "best_position": dyn_tolist(self.pos_best)}}
        )

        # Params
        agent_dict.update(
            {
                "pso": {
                    "inertial_weight": self.w_inertial,
                    "cognitive_weight": self.w_cognitive,
                    "social_weight": self.w_social,
                }
            }
        )

        return agent_dict

    @classmethod
    def from_dict(Agent, d):
        """_summary_

        Args:
            Agent (Agent): _description_
            d (dict): _description_

        Returns:
            Agent: _description_
        """
        return Agent(
            pos_init=dyn_toarray(d["dyn"]["position"]),
            pos_dicts=d["dyn"]["pos_dicts"],
            vel_init=dyn_toarray(d["dyn"]["velocity"]),
            force_init=dyn_toarray(d["dyn"]["force"]),
            nspindim=d["pes"]["num_spindim"],
            w_inertial=d["pso"]["inertial_weight"],
            w_cognitive=d["pso"]["cognitive_weight"],
            w_social=d["pso"]["social_weight"],
            fit=d["fit"]["fitness"],
            fit_best=d["fit"]["best_fitness"],
            pos_best=dyn_toarray(d["fit"]["best_position"]),
        )


##############################################################
# Example potential energy surfaces


# consistent with: https://arxiv.org/pdf/2009.01910.pdf
class HeisenbergModelPES:
    """_summary_
    """
    def __init__(self, j_ex_mats, k_anis, uvecs_anis, h_vec, num_spin, nspindim=3):
        """_summary_

        Args:
            j_ex_mats (_type_): _description_
            k_anis (_type_): _description_
            uvecs_anis (_type_): _description_
            h_vec (_type_): _description_
            num_spin (_type_): _description_
            nspindim (int, optional): _description_. Defaults to 3.
        """
        self.nspindim = nspindim

        # Magnetism model
        self.num_spin = num_spin
        self.j_ex_mats = j_ex_mats
        self.k_anis = k_anis
        self.uvecs_anis = uvecs_anis
        self.h_vec = h_vec

    def evaluate_energy(self, magmoms):
        """_summary_

        Args:
            magmoms (_type_): _description_

        Returns:
            _type_: _description_
        """
        energy = 0.0
        for i in range(self.num_spin):
            for j in range(self.num_spin):
                if i != j:
                    # Jiso, Janis, & DM
                    energy += -np.dot(magmoms[i], np.dot(self.j_ex_mats[i][j], magmoms[j]))
        for i in range(self.num_spin):
            # On-site anis.
            energy += -self.k_anis[i] * np.dot(self.uvecs_anis[i], magmoms[i]) ** 2
            # applied field
            energy += -np.dot(self.h_vec, magmoms[i])

        return energy

    def compute_gradient(self, magmoms):
        """_summary_

        Args:
            magmoms (_type_): _description_

        Returns:
            _type_: _description_
        """
        grads = [0.0 * m for m in magmoms]

        for p in range(self.num_spin):
            energy_p = np.zeros([self.nspindim])
            for i in range(self.num_spin):
                for j in range(self.num_spin):
                    if i != j:
                        # Jiso, Janis, & DM
                        if i == p:
                            for da in range(self.nspindim):
                                for db in range(self.nspindim):
                                    energy_p[da] += -self.j_ex_mats[i][j][da, db] * magmoms[j][db]
                        elif j == p:
                            for da in range(self.nspindim):
                                for db in range(self.nspindim):
                                    energy_p[db] += -magmoms[i][da] * self.j_ex_mats[i][j][da, db]

            for da in range(self.nspindim):
                # On-site anis.
                energy_p[da] += -2.0 * self.k_anis[p] * np.dot(self.uvecs_anis[p], magmoms[p]) * self.uvecs_anis[p][da]
                # applied field
                energy_p[da] += -self.h_vec[da]

            grads[p][:] = energy_p[:]

        return grads
