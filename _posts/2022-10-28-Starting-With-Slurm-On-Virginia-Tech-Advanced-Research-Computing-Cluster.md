---
tags: slurm, beginner
layout: post
---
This quick tutorial is meant for the first time users of [`Slurm`](https://slurm.schedmd.com/documentation.html) for running
jobs/experiments on the [Virginia Tech's Advanced Research Computing](https://arc.vt.edu) (ARC) cluster, hence useful to only the Virginia
Tech's students/employees with access to VT's ARC.

## Introduction
Unlike the local system you would have to run your experiments, deploying code on
VT's ARC clusters is not as straightforward as issuing a simple e.g., `python
"some_file_name.py"` command in your terminal. VT's ARC system uses `Slurm` workload
manager to help its users run their experiments. In simple terms, it is a system
where you request for your desired computing resources and then submit your request
to execute your experiment; the system then schedules your job, and runs it as and
when your requested resources are available. Therefore, at the very minimum, you have to pre-determine the compute resource requirements to execute your code, e.g.,
the number of CPU cores (and/or GPUs), amount of memory, and an expected run time!
Once you have determined these, the next step is to create a `batch` script stating
your resource requirements and the python script you want to run, and then submitting that `batch` script to `Slurm`.

Note that there are two basic ways to execute your code on VT's ARC that I
know of:
* Submitting a `batch` script, and
* `Interact`ive mode

Submitting a `batch` script is recommended when you don't have to constantly interact
with the your code e.g., when training a deep learning model; and `interact`ive mode
is useful when you have to constantly interact with your code e.g., when working with
`Jupyter` notebooks. Following tutorial will help you to quickly set up the `batch`
script for submitting jobs/experiments on `Slurm`, as well as show you how to use
the `interact`ive mode.

## Creating and Submitting a `batch` script
Let's start with logging-in to the VT's ARC, shall we! We do that by doing an
`ssh` to the ARC's login nodes - `tinkercliffs1.arc.vt.edu` and `tinkercliffs2.arc.vt.edu`. Execute the following command in your local machine's terminal.
```
ssh <your_user_id>@tinkercliffs2.arc.vt.edu
```
where `your_user_id` is your User-ID with which you registered yourself on VT's ARC
(by the way, you can log-in to `tinkercliffs1.arc.vt.edu` too). It will then ask you
to enter your VT PID password, followed by approving the pushed Duo login-request on
your phone. Note that it won't prompt on the terminal asking you to approve, so keep
an eye on the Duo notification. Once you are in, you can create your preferred
directories, `git clone` your Github repository, etc. Change to the directory
where your python script is written which you want to run on VT's ARC. There,
create a file named "some\_file\_name.sh", e.g., `run_script.sh`. On the terminal
you can do that by the following command (in the vim editor):
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
* `account` is your Allocation Account's name under which you are registered on ARC.
* `partition` is the queue in which you want to queue in your job. It can be
`normal_q`, `t4_normal_q` (if you want to use the NVIDIA Tesla T4 GPUs), etc.
* `nodes` is the number of nodes you would require to run your job.
* `cpus-per-task` is the number of CPU cores you would require for a smooth running.
* `mem` is the amount of RAM your experiment would need, and finally
* `time` is the amount of real world time your experiment would take to run to its
completion.

In the above example, my sample task will be submitted to a `normal_q` partition
with computational requirements of $$1$$ node and $$1$$ CPU core (on that particular
node), and a maximum $$4$$GB RAM. I expect the task in "some\_file\_name.py" to
run for $$1$$ day and $$12$$ hours. Once done writing your `batch` script, submit
it to the ARC by executing the following command on the terminal:
```
sbatch run_script.sh
```
If the job submission was successful, you can check the status by executing the
following command:
```
squeue -u <your_user_id>
```
That's all there is to it. In the directory from where you submitted your job,
you will also see a new file created with name `slurm-<JOBID>.out` which would
give you the current run time status of your submitted job when you open it. Note
that `<JOBID>` is the ID associated to your submitted job by the `Slurm`. You can
even do a `jobload <JOBID>` on the terminal to see more details of your submitted
job.

## Creating an `Interact`ive job and accessing the Jupyter notebook
As mentioned earlier, you can spawn an `interact`ive job on the ARC and access the
Jupyter notebook to interact with your code in real-time. To do that, log-in to the
VT's ARC cluster (e.g. `tinkercliffs2.arc.vt.edu`) as done above, and execute the
following command in the terminal:
```
interact -A <your_allocation's_account_name> -t 3:00:00 --mem=4G
```
It will start an `interact`ive job under your Allocation Account's name for a period
of $$3$$ hours with $$4$$GB of RAM allotted to your job. Note that the job will
automatically be queued in the `interactive_q` partition with $$1$$ core allotted
to it be default. You can find more details on an `interact`ive job [here](https://www.docs.arc.vt.edu/usage/faq.html#how-do-i-submit-an-interactive-job). Upon the
successful execution of the above command, you will see yourself logged in to a
new compute node (e.g., `tc307` in my case). Next, assuming you have `jupyterlab`
already installed on the login node, execute the following command on the newly
logged-in compute node's terminal to start a `jupyterlab` session with no browser
running.

```
jupyter-lab --no-browser
```
If successful, you will see a `Jupyter` server spawned up with following output
URL on the compute node's terminal (along with some other outputs):
```
http://localhost:8888/lab?token=9b75d001b317496a30cf28c0d768775ff954f2fa059a6491
```
Take note of the `port` number here, which is $$8888$$ and the `token` ID. It will
be useful later. To access this `jupyterlab` session (running on the compute node)
on our local system, we need to do a clever little hack. We will use the `ssh`
tunneling feature to access the session on our local machine via the login node
(i.e. `tinkercliffs2`). Execute the following on your local machine:
```
ssh -L 127.0.0.1:8080:127.0.0.1:8888 <your_user_id>@tinkercliffs2.arc.vt.edu -t ssh -L 8888:localhost:8888 <your_user_id>@tc307
```
You will be required to enter your VT PID password again, followed by approving the
Duo log-in prompt on your phone. If it throws some errors and doesn't ask you for
your password, you may have some mismatch in the `port` numbers; read further,
following info may help you.

Note that there are four `port` numbers that you need to care for in the above
command (first to last): $$8080$$, $$8888$$, $$8888$$, and $$8888$$. The fourth
`port` number i.e. $$8888$$ is from the URL you got on the compute node. If the
`port` number displayed in the URL is different from $$8888$$, e.g., $$8889$$,
then use $$8889$$ as the fourth `port` number. The third `port` number is where
you want to redirect the session on the login node. Note that the third and the
second `port` numbers should be the same. The first `port` number is where you
want to access the session on your local machine. Also note the hostname of the
compute node at the last in the command, it's `tc307` in my case; it could be
different in yours, so replace `tc307` in the command accordingly.

Next up, fire your local browser and enter the URL: `127.0.0.1:8080`. You will see
a `jupyterlab` session coming up which requires a `token` ID to be authenticated.
Enter the `token` ID you got in the URL on your compute node (i.e. on `tc307` in
my case) and you will be able to access the interactive `jupyterlab` session locally! That's it!

## Bonus
I found that on my log-in node, only `python2.7` was present. In case you need
`python3.x`, you can easily install the required version of `python3.x` with
[miniconda](https://docs.conda.io/en/latest/miniconda.html). Once done, you can
set up your desired `python` environment. I have found [venv](https://docs.python.org/3/library/venv.html) to be quite stable, enabling easy installation of `python`
libraries e.g., `jupyterlab`, `numpy`, etc., with `pip`. Note that you need to
`activate` your appropriate `python` environment on the login node and/or the
compute node before you submit jobs or start an interactive session.

Hope this helps you smoothly begin your High Performance Computing journey on VT's
ARC. Feel free to post questions/suggestions and I will try my best to address them
in time. With time, I may keep on updating this tutorial with tips and tricks!

---
