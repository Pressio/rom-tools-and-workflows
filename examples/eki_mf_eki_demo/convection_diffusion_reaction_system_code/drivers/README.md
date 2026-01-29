## Example workflow for building a ROM via RB-greedy and executing it.

Directories:

offline:
	offline_yaml.yaml: settings for construction of the ROM via reduced-basis. 
        run_build_rom_via_greedy.py: script for building ROM via RB greedy. Relies on offline_yaml.yaml, and the cdr.py and cdr_rom.py source files
online_rom:
	input_rom.yaml: settings on how we want to execute a ROM 
        run_rom.py. Script for running a ROM. Relies on input_rom.yaml, the output of the run_build_rom_via_greedy.py script, and the cdr.py and cdr_rom.py source files
online_fom:
	input_fom.yaml: settings on how we want to execute a FOM, mainly will be used to validate the online_rom
        run_fom.py. Script for running a FOM. Relies on input_fom.yaml and the cdr.py source files


A sample workflow would be to go to the offline directory, build a ROM via RB greedy, and then test it with the online_rom driver. 
