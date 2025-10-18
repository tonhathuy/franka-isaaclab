# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/frankaIsaaclab
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

## Available Tasks

This project includes three robotic manipulation tasks using the Franka Emika Panda robot:

### 1. Reach Task

The robot learns to move its end-effector to reach a randomly positioned target point in 3D space.

**Available Environments:**
- `Template-Reach-v0`: Training environment
- `Template-Reach-Play-v0`: Inference/play environment

**Training:**
```bash
python scripts/skrl/train.py --task=Template-Reach-v0
```

**Testing with trained model:**
```bash
python scripts/skrl/play.py --task=Template-Reach-Play-v0 --checkpoint=<path_to_checkpoint>/best_agent.pt
```

**Testing with random actions:**
```bash
python scripts/random_agent.py --task=Template-Reach-v0 --num_envs=4
```

### 2. Lift Task

The robot learns to grasp a cube and lift it to a target height above the table.

**Available Environments:**
- `Template-Lift-v0`: Training environment
- `Template-Lift-Play-v0`: Inference/play environment

**Training:**
```bash
python scripts/skrl/train.py --task=Template-Lift-v0
```

**Testing with trained model:**
```bash
python scripts/skrl/play.py --task=Template-Lift-Play-v0 --checkpoint=<path_to_checkpoint>/best_agent.pt
```

**Testing with random actions:**
```bash
python scripts/random_agent.py --task=Template-Lift-v0 --num_envs=4
```

### 3. Stack Task

The robot learns to stack cubes on top of each other. Two variants are available:

**Available Environments:**
- `Template-Stack-v0`: Standard stacking task
- `Template-Stack-2Cubes-v0`: Simplified 2-cube stacking task (easier to train)

**Training (2-cube variant):**
```bash
python scripts/skrl/train.py --task=Template-Stack-2Cubes-v0
```

**Testing with trained model:**
```bash
python scripts/skrl/play.py --task=Template-Stack-2Cubes-v0 --checkpoint=<path_to_checkpoint>/best_agent.pt --num_envs=4
```

**Training (standard variant):**
```bash
python scripts/skrl/train.py --task=Template-Stack-v0
```

### General Options

All tasks support the following common options:

- `--num_envs`: Number of parallel environments to run (default varies by task)
- `--checkpoint`: Path to a trained checkpoint for inference/play mode
- `--headless`: Run without GUI for faster training

**Example with options:**
```bash
python scripts/skrl/train.py --task=Template-Reach-v0 --num_envs=64 --headless
```

### Monitoring Training

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir logs
```

Training logs are saved in the `logs/skrl/` directory with timestamps.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/frankaIsaaclab/frankaIsaaclab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/frankaIsaaclab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```