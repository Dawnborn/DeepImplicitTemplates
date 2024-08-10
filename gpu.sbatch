#!/bin/bash
#SBATCH --job-name="dditnach_sofaeval"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:TURING"
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user="ge32hij@mytum.de"
#SBATCH --mail-type=END
#SBATCH --output=/home/wiss/lhao/junpeng/sbatchout/logs/slurm-%j.out
#SBATCH --error=/home/wiss/lhao/junpeng/sbatchout/logs/slurm_error-%j.out
#SBATCH --partition="NORMAL"

# Start tmux
# tmux new -d -s "test_optix8"

# Run tmux commands to setup the session
# tmux split-window -h # this will setup two panes, splitting the screen horizontally

# Finally, make job wait for user to connected
# sleep 24h

# # on login node: srun --jobid <JOBID> --pty tmux a -t test_optix8
# srun /home/wiss/lhao/storage/user/hjp/ws_optix/Optixrenderer/buildnode10/bin/optixRenderer -f /home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/0scene0594_00/main_hjp.xml -o /home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/0scene0594_00/finetune_full100/envmap/pfm/bgr.pfm -c /home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/0scene0594_00/finetune_full100/envmap/scene0594_00_control_cam.txt

nvidia-smi
# srun bash train.sh
# srun bash evaluate_reconstruct.bash
srun /usr/wiss/lhao/anaconda3/envs/hjp_deepsdf/bin/python /home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/reconstruct_deep_implicit_templates_generated_latentcode.py
