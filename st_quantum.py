import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft, scf, mp
from pyscf.geomopt import geometric_solver
import numpy as np

# Convert SMILES to XYZ coordinates
def smiles_to_xyz(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = [conf.GetAtomPosition(i) for i in range(len(atoms))]

        xyz = ""
        for atom, coord in zip(atoms, coords):
            xyz += f"{atom} {coord.x:.6f} {coord.y:.6f} {coord.z:.6f}\n"
        return xyz.strip(), mol
    except Exception as e:
        st.error(f"SMILES to XYZ conversion failed: {e}")
        return "", None

# Cached version of run_calculation for efficiency
@st.cache_data
def run_calculation(xyz_string, basis_set, method, optimization, charge, multiplicity, xc_func='b3lyp'):
    mol = gto.Mole()
    mol.atom = xyz_string
    mol.basis = basis_set
    mol.unit = 'Angstrom'
    mol.verbose = 0
    mol.charge = charge
    mol.spin = multiplicity - 1  # PySCF uses 2S
    mol.build()

    mo_energies = None
    dipole_vector = None
    total_energy = None

    try:
        if method == 'DFT':
            mf = dft.RKS(mol)
            mf.xc = xc_func
            if optimization == 'Geometry Optimization':
                mol_optimized = geometric_solver.optimize(mf)
                mf_optimized = dft.RKS(mol_optimized)
                mf_optimized.xc = xc_func
                mf_optimized.kernel()
            else:
                mf.kernel()
                mf_optimized = mf

            if mf_optimized.converged:
                mo_energies = mf_optimized.mo_energy
                dipole_vector = mf_optimized.dip_moment()
                total_energy = mf_optimized.e_tot
            else:
                raise RuntimeError("DFT SCF calculation did not converge.")

        elif method == 'MP2':
            mf = scf.RHF(mol)
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("HF calculation did not converge.")

            if optimization == 'Geometry Optimization':
                mol_optimized = geometric_solver.optimize(mf)
            else:
                mol_optimized = mol

            st.warning("MP2 geometry optimization is performed at the HF level.")

            mp2 = mp.MP2(mf)
            mp2.kernel()

            if mp2.e_corr is not None and not np.isnan(mp2.e_corr):
                mo_energies = mf.mo_energy
                dipole_vector = mf.dip_moment()
                total_energy = mp2.e_tot
            else:
                raise RuntimeError("MP2 correlation energy could not be determined.")

        if mo_energies is not None and dipole_vector is not None:
            homo = max(mo_energies[mo_energies < 0])
            lumo = min(mo_energies[mo_energies > 0])
            dipole = np.linalg.norm(dipole_vector)

            optimized_xyz = None
            if optimization == 'Geometry Optimization':
                atoms = [atom[0] for atom in mol_optimized._atom]
                coords = mol_optimized.atom_coords()
                optimized_xyz = "\n".join(
                    f"{atom} {x:.6f} {y:.6f} {z:.6f}"
                    for atom, (x, y, z) in zip(atoms, coords)
                )

            return {
                'HOMO (eV)': homo * 27.2114,
                'LUMO (eV)': lumo * 27.2114,
                'Dipole Moment (Debye)': dipole,
                'Total Energy (Hartree)': total_energy,
                'Optimized XYZ': optimized_xyz
            }

        else:
            raise RuntimeError("Failed to extract molecular orbital or dipole information.")

    except Exception as e:
        st.error(f"Calculation error: {e}")
        return None

def main():
    st.title("Quantum Chemistry with PySCF")
    st.markdown("Perform DFT or MP2 calculations with or without geometry optimization.")

    use_xyz = st.checkbox("Input geometry manually (XYZ format)")
    if use_xyz:
        xyz = st.text_area("Paste XYZ coordinates", height=150)
        mol_image = None
    else:
        smiles = st.text_input("Enter SMILES string", "CCO")
        xyz, mol = smiles_to_xyz(smiles)

    charge = st.number_input("Molecular Charge", value=0, step=1)
    multiplicity = st.number_input("Spin Multiplicity", value=1, step=1, min_value=1)
    
    if st.checkbox("Show generated XYZ coordinates"):
        st.code(xyz, language="xyz")

    method = st.selectbox("Quantum Method", ["DFT", "MP2"])

    if method == 'DFT':
        xc_func = st.selectbox("Exchange-Correlation Functional", ["b3lyp", "pbe", "m06-2x"])
    else:
        xc_func = None

    basis_set = st.selectbox("Basis Set", [
        "sto-3g", "3-21g", "6-31G", "6-31G*", "6-31G(d)",
        "6-31+G(d)", "6-311G", "6-311G*", "6-311+G(d)", "6-311G(d,p)",
        "cc-pVDZ", "cc-pVTZ"
    ])

    optimization = st.radio("Calculation Type", ["Geometry Optimization", "Single-Point Energy"])

    if method == 'MP2' and optimization == 'Geometry Optimization':
        st.warning("MP2 geometry optimization is not directly supported; optimization is performed at the HF level.")
    
    energy_units = st.radio("Output Energy Units", ["Hartree", "eV"])

    if st.button("Run Calculation"):
        result = run_calculation(xyz, basis_set, method, optimization, charge, multiplicity, xc_func)

        if result:
            st.subheader("Results")
            homo = result['HOMO (eV)']
            lumo = result['LUMO (eV)']
            energy = result['Total Energy (Hartree)']

            if energy_units == "eV":
                energy *= 27.2114
                st.write(f"Total Energy (eV): {energy:.4f}")
            else:
                st.write(f"Total Energy (Hartree): {energy:.6f}")

            st.write(f"HOMO Energy (eV): {homo:.4f}")
            st.write(f"LUMO Energy (eV): {lumo:.4f}")
            st.write(f"Dipole Moment (Debye): {result['Dipole Moment (Debye)']:.4f}")

            if optimization == "Geometry Optimization" and result.get('Optimized XYZ'):
                st.subheader("Optimized Geometry (XYZ)")
                st.text(result['Optimized XYZ'])
                
if __name__ == "__main__":
    main()
