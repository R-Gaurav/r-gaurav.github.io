---
tags: hello
layout: post
---
This quick tutorial is meant for the first time users of `Slurm` for running
jobs/experiments on the Virginia Tech's ARC, hence useful to only the Virginia
Tech's students/employees with access to VT's ARC. Examples are in `python`.

## Introduction
Unlike the local system you would have to run your experiments, deploying code on
VT's ARC clusters is not as straightforward as issuing a simple e.g., `python
"some_file_name.py"` command. VT's ARC system uses `Slurm` workload manager to help
its users run their experiments. In simple terms, you may understand it as a system
where you request for your desired computing resources and then submit your request
to execute your experiment; the system then schedules your job as and when your
requested resources are available. Therefore, at the very minimum, you have to
pre-determine the computing resource requirements to execute your code, e.g.,
the number of CPU cores (and/or GPUs), amount of memory, and an expected run time!
Once you have determined these, the next step is to create a `batch` script stating
your resource requirements and the python script, and then submitting that `batch`
script to `Slurm`.

Note that there are two basic ways to execute your code on VT's ARC that I
know of -
* Submitting a `batch` script, and
* Interactive mode

Submitting a `batch` script is useful when you don't have to constantly interact
with the your code e.g., when training an AI model; and Interactive mode is useful
when you have to constantly interact with your code e.g., when working with `Jupyter`
notebooks. Following tutorial will help you to quickly set up the `batch` script
for Submitting jobs/experiments, as well as show you how to use the Interactive mode.

## Creating and Submitting a `batch` script
Let's start with logging-in to the VT's ARC, shall we!
```
ssh <userid>@tinkercliffs2.arc.vt.edu
```
where `userid` is your User-ID with which you registered yourself on VT's ARC.
By the way, you can log in to `tinkercliffs1.arc.vt.edu` too. Once you enter your
VT PID password, you will have to approve the pushed Duo login-request on your
phone. Note that it won't prompt on the terminal asking you to approve, so keep
an eye on the notification. Once you are in, you can create your preferred
directories, `git clone` your Github repository, etc. Change to the directory
where your python script is written which you want to run on VT's ARC. Create a
file named "some\_file\_name.sh", e.g., `run_script.sh`. On the terminal you can
do that by the following (in the vim editor):
```
vim run_script.sh
```
Type in the following contents in the newly created `run_script.sh` file.
```
#!/bin/bash
#SBATCH --account=<your_allocation's_account_name>
#SbATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-12:00:00
python <some_file_name.py>
```
where:
* `account` is your Allocation Account's name under which you were registered.
* `partition` is the queue in which you want to queue in your job. It can be
`normal_q`, `t4_normal_q` (if you want to use the NVIDIA Tesla T4 GPUs), etc.
* `nodes` is the number of nodes you would require to run your job.
* `cpus-per-task` is the number of CPU cores you would require.
* `mem` is the amount of RAM your experiment woudl need, and finally
* `time` is the amount of real world time your experiment would take to run to its
completion.

In the above example, my sample task is submitted to a `normal_q` with
computational requirements of $1$ node and $1$ CPU core (on that particular node)
, and a maximum $4$GB RAM. I expect the task in "some\_file\_name.py" to run for
$1$ day and $12$ hours. Once done writing your `batch` script, submit it to the
ARC by executing the following command on the terminal:
```
sbatch run_script.sh
```
If the job submission was successful, you can check the status by executing the
following command:
```
squeue -u <userid>
```

That's all there is to it. In the directory from where you submitted your job,
you will also see a new file created with name `slurm-<JOBID>.out` which would
give you the current run time status of your submitted job.

## Creating an Interactive job and accessing Jupyter notebook
As mentioned earlier, you can spawn an Interactive job on the ARC and access the
Jupyter notebook to interact with your code in realtime. To do that, log in to the
VT's ARC as done above, and execute the following command:
```
interact -A <your_allocation's_account_name> -t 3:00:00 --mem=4G
```
It will start an interactive job under your Allocation Account's name for a period
of $3$ hours with $4$GB of RAM available to you. Note that the job will
automatically be queued in the `interactive_q` partition with $1$ core allocated
to it be default. You can find more details on an interactive job [here](https://www.docs.arc.vt.edu/usage/faq.html#how-do-i-submit-an-interactive-job). Upon a
successful execution of the above command, you will see yourself logged in to a
new compute node, e.g., `tc307`. Next, assuming you have `jupyterlab` installed
on the login node (note that the same environment will be present on a compute
node automatically, you **don't** need to set up a new one fresh), execute the
following command to start a `jupyterlab` session with no browser running.

```
jupyter-lab --no-browser
```
If successful, you will see a `Jupyter` server spawned up with following output
on the terminal:
```
http://localhost:8888/lab?token=9b75d001b317496a30cf28c0d768775ff954f2fa059a6491
```
Take note of the `port` here which is $$8888$$ and the `token` ID. We need to do
a clever little hack to access the `jupyterlab` session running on the compute
node we just logged in (i.e. `tc307` in my case).


---
