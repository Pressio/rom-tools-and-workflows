
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 64 64 --outDir mesh_HF
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 32 32 --outDir mesh_LF1
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 16 16 --outDir mesh_LF2
dakota -i dakota_mfmc.in -o dakota.out