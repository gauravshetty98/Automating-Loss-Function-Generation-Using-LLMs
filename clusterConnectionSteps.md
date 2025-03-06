# Connecting to the Cluster and Running Jupyter Notebook

## Step 1: Connect to the Cluster
```bash
ssh gss119@amarel.rutgers.edu
```
- This command establishes a secure shell (SSH) connection to the Amarel research computing cluster at Rutgers University using your username.

## Step 2: Activate Conda Environment
```bash
conda activate te1
```
- This activates the Conda environment named `te1`, ensuring that the necessary dependencies are available for running Jupyter Notebook.

## Step 3: Check Available GPU Resources
```bash
sinfo -s
```
- Displays the status of available partitions (computing resources) on the cluster.

```bash
sinfo --partition=gpu --format="%N %G %T"
```
- Filters and displays the GPU partition details, including nodes, available GPUs, and their current state.

## Step 4: Request GPU Resources
```bash
srun --partition=gpu --gres=gpu:2 --mem=16G --cpus-per-task=1 --time=01:00:00 --pty bash
```
- Requests access to two GPUs, 16GB of memory, and one CPU for one hour in an interactive session.

## Step 5: Verify GPU Availability
```bash
nvidia-smi
```
- Displays the status of allocated GPUs, including memory usage and current workload on each GPU.

## Step 6: Start Jupyter Notebook on Compute Node
```bash
jupyter notebook --no-browser --port=8888
```
- Launches a Jupyter Notebook server on the compute node without opening a browser, binding it to port 8888.

## Step 7: Open a New Terminal and Establish an SSH Tunnel
```bash
ssh -L 8888:localhost:8888 gss119@amarel.rutgers.edu
```
- This command creates an SSH tunnel from your local machine to the cluster, allowing access to Jupyter Notebook via a browser.

## Step 8: Connect to the Compute Node Running Jupyter
```bash
ssh -L 8888:localhost:8888 gpu016
```
- Establishes another SSH tunnel to the specific compute node (`gpu016`) where Jupyter Notebook is running.

## Final Step: Open Jupyter Notebook in Browser
- Open a web browser and go to:
  ```
  http://localhost:8888/?token=<your_token>
  ```
  - Replace `<your_token>` with the actual token displayed in the Jupyter Notebook terminal output.

- This provides access to Jupyter Notebook running on the compute node through your local machine.

