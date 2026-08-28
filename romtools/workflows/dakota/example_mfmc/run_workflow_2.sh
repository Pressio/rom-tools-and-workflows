
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 128 128 --outDir mesh_HF
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 32 32 --outDir mesh_LF1
python3 /home/rloekvh/pressio-demoapps/meshing_scripts/create_full_mesh_for.py \
        --problem diffreac2d -n 16 16 --outDir mesh_LF2

python3 save_model_LF1.py
python3 save_model_LF2.py
python3 save_model_HF.py

dakota -i dakota_mfmc.in -o dakota.out