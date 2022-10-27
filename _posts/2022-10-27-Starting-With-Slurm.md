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
"file_name.py"` command. VT's ARC system uses `Slurm` workload manager to help its
users to run their experiments. At the very minimum, you have to pre-determine the
computing resource requirements to execute your code, e.g., the number of CPU cores
(and/or GPUs), amount of memory, and an expected run time! Once you have determined
these, the next step is to create a `batch` script stating your resource requirements
and the python script, and then submitting that `batch` script to `Slurm`.

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
You can log in to `tinkercliffs1.arc.vt.edu` too. Once you enter your VT PID
password, you will have to approve the pushed Duo login-request on your phone. Note
that it won't prompt on the terminal asking you to approve, so keep an eye on the
notification. Once you are in, you can create your preferred directories, `git clone`
your Github repository, etc. Change to the directory where your python script is
written which you want to run on VT's ARC. Create a file named "some\_file\_name.sh",
e.g. `run_script.sh`. On the terminal you do that by following:
```
vi run_script.sh
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
python <file_name.py>
```
where:
* `account` is

```python
Hello!
```

---
