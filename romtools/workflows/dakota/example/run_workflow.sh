
python3 $PATH_TO_PDA/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 32 32 --outDir mesh_32x32
dakota -i dakota.in -o dakota.out