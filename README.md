# Pig Atrial Tissue Model
This repository includes code relating to a electrophysiological cardiac model of pig atrial tissue, published in _Frontiers in Physiology_, 2022 (https://doi.org/10.3389/fphys.2022.812535).

Scripts correspond to the different protocols used in the manuscript.

-----------------------------------------------------------------------------------------------------

## Restitution Process

Code found in "restitution/". Code runs the protocol from the manuscript in a $8.448\times 0.176 \text{cm}^2$ patch of simulated tissue. Action Potential is recorded, under stimulation at 1Hz, in the center of the tissue and outputted to "output_merged/V_1Hz.csv" (the user must create the directory manually).

To run the code, simply type "mpiexec -n N python main_2D_pulse_train_restitution_ready.py" in the command prompt, with N>1 (if you use a single process, the AP is measured at the edge of the patch).
