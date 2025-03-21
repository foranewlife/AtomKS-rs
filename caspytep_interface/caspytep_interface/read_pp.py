import os
from pathlib import Path

import caspytep
import numpy as np
from ase.atoms import Atoms
from caspytep.otfg import otfg_get, otfg_inquire
from caspytep.otfg_pspot import pspot, otfg_pspot_allocate
from caspytep.otfg_basis import pseudo_basis
from caspytep.otfg_atom import pseudo_atom
from caspytep.otfg_scf import otfg_pseudo_scf_test


def castep_initial(seed="castep_in_a"):
    caspytep.cell.cell_read(seed)

    current_cell = caspytep.cell.get_current_cell()
    caspytep.ion.ion_read()

    real_charge = float(
        np.dot(current_cell.num_ions_in_species, current_cell.ionic_charge)
    )
    real_num_atoms = current_cell.mixture_weight.sum()
    fixed_cell = current_cell.cell_constraints.max() == 0

    caspytep.parameters.parameters_read(seed, real_charge, real_num_atoms, fixed_cell)
    current_params = caspytep.parameters.get_current_params()

    caspytep.comms.comms_parallel_strategy(
        current_params.data_distribution,
        current_params.nbands,
        current_cell.nkpts,
        current_params.num_farms_requested,
        current_params.num_proc_in_smp,
    )

    caspytep.cell.cell_distribute_kpoints()
    caspytep.ion.ion_initialise()

    caspytep.parameters.parameters_output(caspytep.stdout)
    caspytep.cell.cell_output(caspytep.stdout)

    caspytep.basis.basis_initialise(current_params.cut_off_energy)
    # current_basis = caspytep.basis.get_current_basis()
    current_params.fine_gmax = current_params.fine_grid_scale * np.sqrt(
        2.0 * current_params.cut_off_energy
    )
    caspytep.ion.ion_real_initialise()

    main_model = caspytep.model.model_state()
    return main_model


def write_cell(ions, outfile, cell_params=None):
    import ase.io.castep as ase_io_driver
    from ase.calculators.castep import Castep as ase_calc_driver

    # ase_atoms = ase_io.ions2ase(ions)
    ase_atoms = ions
    ase_atoms.set_calculator(ase_calc_driver())
    ase_cell = ase_atoms.calc.cell
    if cell_params is not None:
        for k1, v1 in cell_params.items():
            if isinstance(v1, dict):
                value = []
                for k2, v2 in v1.items():
                    value.append((k2, v2))
                ase_cell.__setattr__(k1, value)
            else:
                ase_cell.__setattr__(k1, v1)
    ase_io_driver.write_castep_cell(outfile, ase_atoms, force_write=True)


def get_core_density(ions, pplist, kgrid="1 1 1", xc="pbe", outfile="core.list"):
    cell_params = {"species_pot": pplist, "kpoints_mp_grid": kgrid}
    prefix = "castep_in_a"
    with open(prefix + ".param", "w") as fh:
        fh.write(
            f" TASK: Energy\n CUT_OFF_ENERGY: 600.0 eV\n XC_FUNCTIONAL: {xc.upper()}\n "
        )
    write_cell(ions, outfile=prefix + ".cell", cell_params=cell_params)
    castep_initial(seed=prefix)

    atomic_radial_charge = caspytep.ion.get_array_atomic_radial_charge()
    ps_num_points = caspytep.ion.get_array_ps_num_points()
    ps_gmax = caspytep.ion.get_array_ps_gmax()
    for i in range(len(ps_num_points)):
        atomic_radial_grid = np.linspace(0, ps_gmax[i], ps_num_points[i])
        np.savetxt(outfile, np.c_[atomic_radial_grid, atomic_radial_charge[:, i]])
        break


def get_pp_pb_pa(
    element: str,
    path: str,
    xc="pbe",
    kgrid="1 1 1",
    spin_orbit_coupling=False,
    density_outfile="core.list",
):
    suffix = Path(path).suffixes
    ppf = Path(path).name

    ions = Atoms([element], [[0, 0, 0]], cell=[3, 3, 3])
    pplist = {element.capitalize(): ppf}

    get_core_density(ions, pplist, kgrid=kgrid, xc=xc, outfile=density_outfile)
    pseudopotential = get_pseudopotentia_core(suffix, spin_orbit_coupling)
    pseudo_basis_instance, pseudo_atom_instance = generate_pseudo_basis_and_atom(
        pseudopotential
    )
    return pseudopotential, pseudo_basis_instance, pseudo_atom_instance


def get_pseudopotentia_core(suffix, spin_orbit_coupling=False):

    (
        tot_num_projectors,
        tot_num_points,
        ionic_charge,
        num_mesh,
        num_kkbeta,
        num_nqf,
        bl_points,
        tot_num_bl_projectors,
        tot_num_core_projectors,
    ) = otfg_inquire(suffix, 1, spin_orbit_coupling)
    pseudopotential = pspot()
    otfg_pspot_allocate(
        caspytep.ion.get_ps_max_points(),
        caspytep.ion.get_max_ps_projectors(),
        caspytep.ion.get_max_bl_projectors(),
        3,
        caspytep.ion.get_max_mesh(),
        caspytep.ion.get_max_kkbeta(),
        num_nqf,
        caspytep.ion.get_max_core_projectors(),
        pseudopotential,
    )
    otfg_get(suffix, 1, pseudopotential)

    return pseudopotential


def generate_pseudo_basis_and_atom(pseudopotential):
    pseudo_basis_instance = pseudo_basis()
    pseudo_atom_instance = pseudo_atom()
    otfg_pseudo_scf_test(pseudopotential, pseudo_basis_instance, pseudo_atom_instance)
    return pseudo_basis_instance, pseudo_atom_instance
