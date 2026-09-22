# MACE-MH-1 thermal-property setup for LAMMPS ML-IAP.
#
# Required environment variables:
#   THERMAL_MACE_MODEL     converted *mliap_lammps.pt model
#   THERMAL_MACE_ELEMENTS element symbols ordered by LAMMPS atom type
#
# This file is intentionally self-contained.  It does not import runtime files
# from SingleSnapshot, and the classical topology is not evaluated in addition
# to the MACE potential.

variable thermal_mace_model getenv THERMAL_MACE_MODEL
variable thermal_mace_elements getenv THERMAL_MACE_ELEMENTS

print "THERMAL_MACE_SETUP_VERSION=thermal_schema_v1_mh1_mliap"
print "THERMAL_MACE_MODEL=${thermal_mace_model}"
print "THERMAL_MACE_ELEMENTS=${thermal_mace_elements}"

pair_style mliap unified ${thermal_mace_model} 0
pair_coeff * * ${thermal_mace_elements}

# Command-line index variables override this fallback. Use a thermal-specific
# name: older launchers pass an unused mace_neigh_skin=0.0, while their actual
# thermal behavior has always been 2.0 A. Direct includes keep that default too.
variable thermal_mace_neigh_skin index 2.0
print "THERMAL_MACE_NEIGH_SKIN_A=${thermal_mace_neigh_skin}"
neighbor ${thermal_mace_neigh_skin} bin
neigh_modify delay 0 every 1 check yes

kspace_style none
bond_style none
angle_style none
dihedral_style none
improper_style none
special_bonds lj/coul 1.0 1.0 1.0
