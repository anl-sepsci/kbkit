"""Create a dummy script for NaCl/water example.

Running electrolyte systems is the same as other systems, except a `charges` attribute is required, mapping residues to their charges. If a residue is not specified, the charge is assumed to be neutral.
"""
from kbkit.api import Pipeline

# get paths to system directories
pure_component_path = '/path/to/pure/components/'
system_set_path = '/path/to/salt/systems/parent/'

# create pipeline
pipe = Pipeline(
    pure_path=pure_component_path,
    base_path=system_set_path,
    pure_systems=[f"{pure_component_path}/NaCl_300", f"{pure_component_path}/SPCEW_300"],
    rdf_dir="kbi_rdf_files",
    start_time=10000, # ignore first 10 ns for property averaging
    include_mode="npt",
    errors="warn",
    charges={"NA": 1, "CL": -1} # REQUIRED FOR ELECTROLYTES (map residue names to their charge)
)

results = pipe.results

# now for plotting KBI's

# note: salt nomenclature = Cation.Anion
salt_idx = pipe.systems.get_mol_index("NA.CL")
x_salt = pipe.systems.x[:,salt_idx]

pipe.kbi_plotter.plot_composition(
    x=x_salt,
    molecules=['NaCl', 'water'],
    xlab=r'$x_{NaCl}$',
    cmap='rainbow',
    marker='o',
    ls='-',
    lw=1,
    mew=1,
    show=True
)

# plotting activity coefficients
pipe.thermo_plotter.plot_property(
    "lngamma", ylabel=r"$\ln \gamma_i$", xmol="NA.CL", lw=1, marker="o", mew=1, cmap="winter", show=True
)

# plotting contributions to mixing free energy ---- note this requires the pure-component simulations
pipe.thermo_plotter.plot_binary_mixing(
    xmol='NA.CL', cmap='rainbow', show=True
)



print('Electrolyte analysis complete!')