# Connecting to the Cluster and Running Jupyter Notebook

These are some of the steps I took to connect with the Amarel cluster and use GPUs to run ML task. In this particular example, I wanted to run my python code on Jupyter Notebook but in my local machine

Prerequiste: A virtual environment with jupyter, torch and other necessary packages should be setup in the cluster. This will be used later to launch the jupyter notebook.

## Step 1: Connect to the Cluster
```bash
ssh <net_id>@amarel.rutgers.edu
```
- This command establishes a secure shell (SSH) connection to the Amarel research computing cluster at Rutgers University using your username.


## Step 2: Check Available GPU Resources
```bash
sinfo -s
```
- Displays the status of available partitions (computing resources) on the cluster.

```bash
sinfo --partition=gpu --format="%N %G %T"
```
- Filters and displays the GPU partition details, including nodes, available GPUs, and their current state.

## Step 3: Request GPU Resources
```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --cpus-per-task=1 --time=01:00:00 --pty bash
```
- Requests access to one GPU, 16GB of memory, and one CPU for one hour in an interactive session.
- Take note of the node that is assigned to you <net_id>@<node_name>

## Step 4: Verify GPU Availability
```bash
nvidia-smi
```
- Displays the status of allocated GPUs, including memory usage and current workload on each GPU.
- If GPUs are assigned properly this should run without error

## Step 5: Activate your virtual environment with the loaded models
```bash
conda activate <venv_name>
```
- This requires you to already have an envrionment with jupyter packages already installed or use ```module load``` command to load anaconda packages.

## Step 6: Start Jupyter Notebook on Compute Node
```bash
jupyter notebook --no-browser --port=8888
```
- Launches a Jupyter Notebook server on the compute node without opening a browser, binding it to port 8888.

## Step 7: Open a New Terminal and Establish an SSH Tunnel
```bash
ssh -L 8888:localhost:8888 <net_id>@amarel.rutgers.edu
```
- This command creates an SSH tunnel from your local machine to the cluster, allowing access to Jupyter Notebook via a browser.

## Step 8: Connect to the Compute Node Running Jupyter
```bash
ssh -L 8888:localhost:8888 <node_name>
```
- Establishes another SSH tunnel to the specific compute node (`<node_name>`) where Jupyter Notebook is running. This is the same node name which you receive after step 3.

## Final Step: Open Jupyter Notebook in Browser
- Open a web browser and go to:
  ```
  http://localhost:8888/?token=<your_token>
  ```
  - Replace `<your_token>` with the actual token displayed in the Jupyter Notebook terminal output.

- This provides access to Jupyter Notebook running on the compute node through your local machine.

