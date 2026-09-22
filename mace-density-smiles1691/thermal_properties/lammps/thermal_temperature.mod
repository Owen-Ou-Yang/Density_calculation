# Shared thermodynamic computes for thermal-property trajectories.
# temp/com removes center-of-mass translation consistently from thermostat,
# pressure, kinetic energy, and total energy reporting.

compute thermal_temperature all temp/com
compute_modify thermal_temperature dynamic/dof yes
compute thermal_pressure all pressure thermal_temperature

variable thermal_force_magnitude atom sqrt(fx*fx+fy*fy+fz*fz)
compute thermal_force_max all reduce max v_thermal_force_magnitude

# Equal-style variables are evaluated lazily by fix print.  The corresponding
# CSV columns therefore contain instantaneous, unaveraged samples suitable for
# autocorrelation-aware post-processing.
variable thermal_csv_step equal step
variable thermal_csv_time_ps equal time
variable thermal_csv_temp_k equal temp
variable thermal_csv_press_bar equal press
variable thermal_csv_density_g_cm3 equal density
variable thermal_csv_volume_a3 equal vol
variable thermal_csv_pe_ev equal pe
variable thermal_csv_ke_ev equal ke
variable thermal_csv_etotal_ev equal etotal
variable thermal_csv_enthalpy_ev equal enthalpy
variable thermal_csv_pxx_bar equal pxx
variable thermal_csv_pyy_bar equal pyy
variable thermal_csv_pzz_bar equal pzz
variable thermal_csv_pxy_bar equal pxy
variable thermal_csv_pxz_bar equal pxz
variable thermal_csv_pyz_bar equal pyz
variable thermal_csv_lx_a equal lx
variable thermal_csv_ly_a equal ly
variable thermal_csv_lz_a equal lz
variable thermal_csv_fmax_ev_a equal c_thermal_force_max
variable thermal_csv_atom_count equal count(all)
