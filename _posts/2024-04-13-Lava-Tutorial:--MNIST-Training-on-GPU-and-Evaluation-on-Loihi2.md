---
tags: tutorial slayer lava loihi
layout: post
---
In this long end-to-end tutorial, I will demonstrate how to train a Spiking Neural Network (SNN) on GPU for MNIST digits classification using [Lava](https://lava-nc.org/) library, and then how to evaluate the same on Loihi-2 neuromorphic chip (both -- **simulation** and **physical** hardware on [INRC](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview)).

`Note`: Complete source code for this tutorial can be found on my [mnist-on-loihi](https://github.com/R-Gaurav/mnist-on-loihi) GitHub repository. Also, basic knowledge of Lava's `Process` and `ProcessModel` is necessary to follow this tutorial well; however, you are welcome to read it as I will be providing high-level explanations of the realted Lava APIs. You can learn more about Lava on its above-linked website.

# Introduction
Programming in Lava is _not_ easy! -- well... at least as of this date. Lava is a rich python library introduced by Intel to build, train, and deploy spiking networks on Intel's neuromorphic chips: $$\textsf{Loihi-1}$$ and $$\textsf{Loihi-2}$$. As of now, Lava supports [Deep Learning through SNNs](https://lava-nc.org/lava-lib-dl/index.html), dynamic spiking networks based on [Dynamic Neural Fields](https://lava-nc.org/lava-lib-dnf/lava.lib.dnf.html), and interestingly - mathematical [Constraint Optimization](https://lava-nc.org/lava-lib-optimization/lava.lib.optimization.html). There are **two** main ways to build and train Deep-SNNs in Lava through [Lava-DL](https://github.com/lava-nc/lava-dl):
* (a) **ANN-to-SNN** via [Bootstrap](https://lava-nc.org/lava-lib-dl/bootstrap/index.html), and
* (b) **Direct Training** via [SLAYER](https://lava-nc.org/lava-lib-dl/slayer/index.html)

You can find more details and tutorials [here](https://lava-nc.org/dl.html). Note that after you have trained your Lava-based Deep-SNN, you can use Lava's [NetX](https://lava-nc.org/lava-lib-dl/netx/index.html) module to port your network to Loihi neuromorphic chips.

>`We here focus on building, (direct) training, and deploying an SNN composed of only Dense layers via Lava's SLAYER and NetX APIs; training on GPU, evaluation on Loihi-2.`

`Note`: Here I am just presenting the main excerpts from the [mnist-on-loihi](https://github.com/R-Gaurav/mnist-on-loihi) repository necessary to understand my code. The library requirements and execution process/commands are mentioned in my repo's README. Also, needless to say, the Loihi-2 chip's simulation (henceforth, $$\textsf{Loihi-Sim}$$) runs on your CPU, while the Loihi-2 chip's actual hardware (henceforth, $$\textsf{Loihi-Hw}$$) can be accessed on the INRC cloud.

I am next introducing some very foundational details of Lava, related to this tutorial.

## Lava's `Process`
`Process`es in Lava define the **Interface** of your network's components. A `Process` consists of one or more **input** and **output** ports through which it communicates with other `Process`es; it also consists of internal variables that it uses for its operations.

## Lava's `ProcessModel`
`ProcessModel`s in Lava are simply the **Implementation** of the corresponing `Process`es on different hardware architectures/backends, e.g., host CPU, Loihi-embedded LMT x86-cores, etc. Each `Process` can have one or more `ProcessModel`s suited for different hardware, and it consists of regular _python_ or _C_ code depending upon the hardware substrate it is intended to run on. Note that a `ProcessModel` follows a certain protocol in agreement of which it executes its operations. All the `ProcessModel`s in this tutorial follow the `LoihiProtocol` that consists of mulitple execution **phases** -- each of the phases follow a specific order; more details on `LoihiProtocol` can be found [here](https://lava-nc.org/lava/lava.magma.core.sync.protocols.html#lava.magma.core.sync.protocols.loihi_protocol.LoihiProtocol).

## Lava's `slayer` Module
The `slayer` module is part of the [Lava-DL](https://lava-nc.org/dl.html) library underneath which runs PyTorch. The `slayer` module offers APIs such as $$\texttt{slayer.utils.Assistant}$$ that assists in training and evaluation of a `SLAYER` SNN, and $$\texttt{slayer.utils.LearningStats}$$ that outputs the training and test metrics/stats.

##  Lava's `netx` Module
The `netx` module is part of the [Lava-DL](https://lava-nc.org/dl.html) library that supports loading/porting the SLAYER-trained SNNs and their deployment on Loihi hardware. Note that the `netx`-loaded networks have 8-bit quantized integer weights.

# Repository Tree
Let's now take a brief glimpse at my [mnist-on-loihi](https://github.com/R-Gaurav/mnist-on-loihi) repository -- I am describing its contents below:

---
* [$$\texttt{./net_utils/utils.py}$$](https://github.com/R-Gaurav/mnist-on-loihi/blob/main/net_utils/utils.py): contains the utility classes --
    * $$\texttt{ExpDataset}$$: contains code for Rate-encoding the MNIST images to spikes - to be fed to $$\texttt{SlayerDenseSNN}$$
    * $$\texttt{InpImgToSpk}$$ and $$\texttt{PyInpImgToSpkModel}$$: contains the code for Rate-encoding the MNIST images to **input** spikes -- to be fed to the network on Loihi, where
        * `Process` is: $$\texttt{InpImgToSpk}$$, and its
        * `ProcessModel` is: $$\texttt{PyInpImgToSpkModel}$$
    * $$\texttt{OutSpkToCls}$$ and $$\texttt{PyOutSpkToClsModel}$$: contains the code for inferring predicted classes from the **output** spikes -- collected from the network on Loihi, where
        * `Process` is: $$\texttt{OutSpkToCls}$$, and its
        * `ProcessModel` is: $$\texttt{PyOutSpkToClsModel}$$
    * $$\texttt{InputAdapter}$$, $$\texttt{PyInputAdapter}$$, and $$\texttt{NxInputAdapter}$$: contains the code to _adapt_ the input spikes from $$\texttt{InpImgToSpk}$$ (on CPU) to the network on Loihi, where
        * `Process` is: $$\texttt{InputAdapter}$$, and its
        * `ProcessModel`s are: $$\texttt{PyInputAdapter}$$ and $$\texttt{NxInputApdater}$$
    * $$\texttt{OutputAdapter}$$, $$\texttt{PyOutputAdapter}$$, and $$\texttt{NxOutputAdapter}$$: contains the code to _adapt_ the output spikes from the network on Loihi to $$\texttt{OutSpkToCls}$$ (on CPU), where
        * `Process` is: $$\texttt{OutputAdapter}$$, and its
        * `ProcessModel`s are: $$\texttt{PyOutputAdapter}$$ and $$\texttt{NxOutputAdapter}$$

* [$$\texttt{./net_utils/snns.py}$$](https://github.com/R-Gaurav/mnist-on-loihi/blob/main/net_utils/snns.py): contains two network classes --
    * $$\texttt{SlayerDenseSNN}$$: contains the code for creating a `Dense`-only SNN using SLAYER (i.e., `slayer`) APIs, and
    * $$\texttt{LavaDenseSNN}$$: contains the code for creating a Loihi-deployable network composed of Lava `Process`es (from $$\texttt{utils.py}$$ above) and the SLAYER-trained network (loaded via `netx`)

* [$$\texttt{./train_eval_snn.py}$$](https://github.com/R-Gaurav/mnist-on-loihi/blob/main/train_eval_snn.py): contains the code to train and evaluate the $$\texttt{SlayerDenseSNN}$$ (on $$\textsf{GPU}$$), as well as evaluate the $$\texttt{LavaDenseSNN}$$ on $$\textsf{Loihi-Sim}$$ and $$\textsf{Loihi-Hw}$$

* [$$\texttt{./trained_mnist_network.pt}$$](https://github.com/R-Gaurav/mnist-on-loihi/blob/main/trained_mnist_network.pt) and [$$\texttt{trained_mnist_network.net}$$](https://github.com/R-Gaurav/mnist-on-loihi/blob/main/trained_mnist_network.net): contains the trained weights/configs of the $$\texttt{SlayerDenseSNN}$$

---

# My Networks Details
The idea is to first build a `Dense`-only SNN (using `slayer` APIs) composed of two **Hidden** layers (with $$128$$ and $$64$$ spiking neurons respectively) and one **Output** layer (with $$10$$ spiking neurons), followed by its training on a $$\textsf{GPU}$$ (again using `slayer`), and finally its evaluation on the $$\textsf{Loihi-Sim}$$ (on CPU) using `netx` and Lava `Process`es. If you have access to the physical $$\textsf{Loihi-Hw}$$ on INRC, the trained `Dense`-only SNN can be evaluated there as well. Note that since my network's input is composed of `Dense` layer, the MNIST images have to be flattened (already taken care by Lava).

## Networks Architecture
* On $$\textsf{GPU}:$$

$$\texttt{ExpDataset} \rightarrow \texttt{SlayerDenseSNN}$$

where $$\texttt{SlayerDenseSNN}$$ implies the network:

$$\texttt{Dense CUBA(128)} \rightarrow \texttt{Dense CUBA(64)} \rightarrow \texttt{Dense CUBA(10)}$$

with $$\texttt{Dense CUBA(m)}$$ denoting fully connected `Dense` connections with '$$\texttt{m}$$' number of $$\texttt{CUBA}$$ post-synaptic neurons.

* On $$\textsf{Loihi-Sim}$$ and $$\textsf{Loihi-Hw}:$$

$$\texttt{InpImgToSpk} \rightarrow \texttt{InputAdapter} \rightarrow \texttt{netx-obtained Network} \rightarrow \texttt{OutputAdapter} \rightarrow \texttt{OutSpkToCls}$$

where the $$\texttt{netx-obtained Network}$$ is simply the trained $$\texttt{SlayerDenseSNN}$$ loaded via the `netx` module. There is one more nuance here, the ground-truth label **Output** port of $$\texttt{InpImgToSpk}$$ `Process` is _directly_ connected to the ground-truth label **Input** port of $$\texttt{OutSpkToCls}$$ `Process`.

Note that for the $$\textsf{Loihi-Sim}$$ backend, I am using the `ProcessModel`s: $$\texttt{PyInpImgToSpkModel}, \texttt{PyOutSpkToClsModel}, \texttt{PyInputAdapter},$$ and $$\texttt{PyOutputAdapter}$$ -- all of which run on CPU; the `netx`-obtained network too, runs on CPU.

Similarly, for the $$\textsf{Loihi-Hw}$$ backend, I am using the `ProcessModel`s: $$\texttt{PyInpImgToSpkModel}$$ and $$\texttt{PyOutSpkToClsModel}$$ that run on CPU, because they generate input spikes and collect output spikes (for inference) respectively; however, to interface with the `netx`-obtained network, which in this case runs on the Loihi neurocores, I am using the `ProcessModel`s: $$\texttt{NxInputAdapter}$$ and $$\texttt{NxOutputAdapter}$$ (both run on Loihi neuro-cores as well) to transfer the spikes to-and-fro between the Loihi neuro-cores and the CPU.

# $$\texttt{SlayerDenseSNN}$$ related Code in Details

I am next explaining my code in details, as well as the related nuances of Lava. It is important to note that the SNNs in Lava are executed for a certain (desired) number of time-steps, starting from time-step $$=1$$ (and **not** $$0$$)! This nuance has some pretty strong implications as we will see later. Let us start with the $$\texttt{SlayerDenseSNN}$$ related code.


## $$\texttt{ExpDataset}$$ Class
This class is fairly easy to understand, where I use the [Nengo/NEF](https://www.nengo.ai/) defined way to Rate-encode the image pixels to binary spikes via the equation:

$$J = \alpha\times<e.x> + \beta$$

where $$J$$ is the input current to the encoding neuron, $$\alpha$$ and $$\beta$$ are its `gain` and `bias` values. Note that $$x$$ is the normalized pixel value to be encoded, and since it's non-negative, the value of the encoder $$e$$ is kept $$+1$$ here ($$<e.x>$$ denotes **dot** product); $$\alpha$$ and $$\beta$$ are set to $$1$$ and $$0$$ respectively. The generated spike trains are fed to the `SlayerDenseSNN`'s first **Hidden** layer.

## $$\texttt{SlayerDenseSNN}$$ Class
The $$\texttt{SlayerDenseSNN}$$ is composed of following `Dense` blocks of Current Based (CUBA) neurons, with their $$\texttt{neuron_params}$$ as described below (code in $$\texttt{snns.py}$$):
```python
neuron_params = {
    "threshold": 1.0,
    "current_decay": 0.10,
    "voltage_decay": 0.10,
    "requires_grad": False,
    }

self.blocks = torch.nn.ModuleList([
    # First Hidden Layer.
    slayer.block.cuba.Dense(
        neuron_params, 784, 128, weight_norm=False, delay=False),
    # Second Hidden Layer.
    slayer.block.cuba.Dense(
        neuron_params, 128, 64, weight_norm=False, delay=False),
    # Output Layer.
    slayer.block.cuba.Dense(
        neuron_params, 64, 10, weight_norm=False, delay=False)
    ])
```
Note that the hidden `Dense` layers have $$128$$ and $$64$$ CUBA neurons; feel free to adjust these parameters as well as the $$\texttt{neuron_params}$$ to improve the accuracy reported here.

## Training and Evaluating $$\texttt{SlayerDenseSNN}$$
Training and evaluating SNNs using `slayer` is quite straightforward. Following code (in $$\texttt{train_eval_snn.py}$$) defines the $$\texttt{loss}$$ function, $$\texttt{stats}$$ monitor, $$\texttt{optimizer}$$, and the training/evaluation $$\texttt{assistant}$$ for the $$\texttt{SlayerDenseSNN}$$:
```python
loss = slayer.loss.SpikeRate(
    # `true_rate` and `false_rate` should be between [0, 1].
    true_rate=0.9, # Keep `true_rate` high for quicker learning.
    false_rate=0.01, # Keep `false_rate` low for quicker learning.
    reduction="sum").to(self.device)

stats = slayer.utils.LearningStats()

optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

assistant = slayer.utils.Assistant(
    self.model, loss, optimizer, stats,
    classifier=slayer.classifier.Rate.predict)
```
Within each training epoch, call the $$\texttt{assistant.train(.)}$$ and $$\texttt{assistant.test(.)}$$ functions to train and evaluate the $$\texttt{SlayerDenseSNN}$$, respectively.

# $$\texttt{LavaDenseSNN}$$ related Code in Details
Given that the training/evaluation of $$\texttt{SlayerDenseSNN}$$ on $$\textsf{GPU}$$ is clear, let's now look into the code of $$\texttt{LavaDenseSNN}$$ (composed of `Process`es and `netx`-obtained network) in details. Following code builds the $$\texttt{LavaDenseSNN}$$ (code in $$\texttt{snns.py}$$):
```python
# -- Lava Trained SNN portable to Loihi (either H/W or Simulation).
self.net = netx.hdf5.Network(
    net_config=trnd_net_path, # Trained network path.
    reset_interval=n_tsteps, # Presentation time-steps of each test-image.
    reset_offset=1 # Phase shift / offset time-step to reset this network.
    )

# Connect Processes.
self.img_to_spk.spk_out.connect(self.inp_adp.inp)
self.inp_adp.out.connect(self.net.inp)
self.net.out.connect(self.out_adp.inp)
self.out_adp.out.connect(self.spk_to_cls.spikes_in)

# Connect ImgToSpk Input directly to SpkToCls Output for ground truths.
self.img_to_spk.label_out.connect(self.spk_to_cls.label_in)
```
where $$\texttt{self.img_to_spk}$$ and $$\texttt{self.spk_to_cls}$$ are the instances of the $$\texttt{InpImgToSpk}$$ and $$\texttt{OutSpkToCls}$$ `Process`es respectively, $$\texttt{self.inp_adp}$$ and $$\texttt{self.otp_adp}$$ are the instances of the $$\texttt{InputAdapter}$$ and $$\texttt{OutputAdapter}$$ `Process`es respectively, and $$\texttt{self.net}$$ is the `netx`-obtained (trained) $$\texttt{SlayerDenseSNN}$$ network.

With respect to the above `netx`-obtained network, note that the $$\texttt{trnd_net_path}$$ is actually the saved (trained) $$\texttt{SlayerDenseSNN}$$ model and $$\texttt{reset_interval}$$ is equal to the presentation time-steps of an image. The $$\texttt{reset_interval}$$ denotes the number of time-steps after which the neurons in the `netx`-obtained network should be reset for a fresh input. Also, $$\texttt{reset_offset}$$ is set to $$1$$ because I want the `netx`-obtained network to reset after every $$\texttt{reset_interval}$$ with a positive offset of $$1$$ time-step (i.e., the phase-shift of the $$\texttt{reset_interval}$$ is $$1$$) -- reasons will be explained later.

## $$\texttt{InpImgToSpk}$$ and $$\texttt{PyInpImgToSpkModel}$$ Classes
The `Process`: `InpImgToSpike` and its corresponding `ProcessModel`: `PyInpImgToSpkModel` implement the code to Rate-encode the normalized pixel values to binary spikes - in a fashion similar to the `ExpDataset` class; the **input** spikes are fed to the `netx`-obtained network's **Input** interface.

The following code (in $$\texttt{utils.py}$$):

```python
@implements(proc=InpImgToSpk, protocol=LoihiProtocol)
@requires(CPU)
class PyInpImgToSpkModel(PyLoihiProcessModel):
```
associates the `Process` - $$\texttt{InpImgToSpk}$$ to its `ProcessModel` - $$\texttt{PyInpImgToSpkModel}$$ and instructs it to execute according to the `LoihiProtocol` on the host `CPU`. Note that as per the `LoihiProtocol`, the constituent `run_spk()` phase/function is executed **first**, followed by the `post_guard()` and `run_post_mgmt()` phases. Note that the $$\texttt{run_post_mgmt()}$$ phase is executed **only** when the $$\texttt{post_guard()}$$ phase returns $$\texttt{True}$$, while the $$\texttt{run_spk()}$$ phase executes every time-step unchecked/unconditionally.
### Components of $$\texttt{PyInpImgToSpkModel}$$
Following is the code of $$\texttt{run_spk()}$$ (in $$\texttt{utils.py}$$):
```python
def run_spk(self):
    if self.time_step % self.n_ts == 1:
        self.inp_img = np.zeros(self.inp_img.shape, dtype=float)
        self.v = np.zeros(self.v.shape, dtype=float)

    J = self.gain*self.inp_img + self.bias
    self.v[:] = self.v[:] + J[:]
    mask = self.v > self.vth
    self.v[mask] = 0
    self.spk_out.send(mask)
```
where $$\texttt{self.n_ts}$$ is the presentation time-steps of an image, $$\texttt{self.inp_img}$$ is the input image, and $$\texttt{self.v}$$ is the neurons' voltage which Rate-encode the input image to spikes. We will come back to the $$\texttt{run_spk()}$$ again. For now, let's look into the code of $$\texttt{post_guard()}$$ next:
```python
def post_guard(self):
    if self.time_step % self.n_ts == 1: # n_ts steps passed, one image processed.
        return True

    return False
```
Above code implies that whenever $$\texttt{self.time_step}$$ is $$1$$ more than a multiple of $$\texttt{self.n_ts}$$, the $$\texttt{post_guard()}$$ phase returns $$\texttt{True}$$, i.e., the $$\texttt{run_post_mgmt()}$$ phase will be executed in that time-step. In other words, $$\texttt{post_guard()}$$ returns $$\texttt{True}$$ when one input image/sample has been presented to the network for the desired number of $$\texttt{self.n_ts}$$ time-steps, such that, the following code in $$\texttt{run_post_mgmt()}$$ will be executed:
```python
def run_post_mgmt(self):
    img = self.mnist_dset.test_images[self.curr_img_id]
    self.inp_img = img/255
    self.ground_truth_label = self.mnist_dset.test_labels[self.curr_img_id]
    self.label_out.send(np.array([self.ground_truth_label]))
    self.v = np.zeros(self.v.shape, dtype=float)
    self.curr_img_id += 1
```
As can be seen above, the input image $$\texttt{self.inp_img}$$ and the corresponding ground truth $$\texttt{self.ground_truth_label}$$ are updated as per the current image index $$\texttt{self.curr_img_id}$$, and the encoding neurons' voltage $$\texttt{self.v}$$ is reset to zeros -- thus readying for the next iteration of $$\texttt{run_spk()}$$ (i.e., Rate-encoding the new updated image) for the next $$\texttt{self.n_ts}$$ time-steps. Finally, $$\texttt{self.curr_img_id}$$ is updated to its next value - to be used in the next execution of $$\texttt{run_post_mgmt()}$$.

Now that we have the component phases of $$\texttt{PyInpImgToSpkModel}$$ `ProcessModel` ready, let's closely look at their operations in tandem.

### Holistic Operation of $$\texttt{InpImgToSpk}$$
Before we begin to understand the `Process` $$\texttt{InpImgToSpk}$$, let's assume that the values of $$\texttt{self.curr_img_id}$$ and $$\texttt{self.n_ts}$$ are $$0$$ and $$20$$ respectively.

* $$\texttt{self.time_step}=1$$:

When the class $$\texttt{InpImgToSpk}$$ is initialized, the $$\texttt{self.inp_img}$$ and $$\texttt{self.v}$$ arrays are all set to zeros. Next, when it is run, the $$\texttt{self.time_step}$$ starts at $$1$$ and $$\texttt{run_spk()}$$ is the first phase to be executed - thus, it updates the encoding neurons' $$\texttt{self.v}$$, but since $$\texttt{self.imp_img}$$ is all zeros (with $$\texttt{self.gain}=1$$ and $$\texttt{self.bias}=0$$), $$\texttt{self.v}$$ remains all zeros. Thus _no_ spikes are generated and sent. After the $$\texttt{run_spk()}$$ phase is over, the $$\texttt{post_guard()}$$ phase is invoked. Keep in mind that $$\texttt{self.time_step}$$ is still $$1$$. Now, the line $$\texttt{if self.time_step % self.n_ts == 1} \implies \texttt{1 % 20}$$, which is equal to $$1$$, thus, $$\texttt{post_guard()}$$ returns $$\texttt{True}$$ and the $$\texttt{run_post_mgmt()}$$ phase is invoked. Note that $$\texttt{self.time_step}$$ is still $$1$$.

In the $$\texttt{run_post_mgmt()}$$ phase, $$\texttt{self.inp_img}$$ is updated with the normalized pixel values of the first test-image, i.e., at the index $$\texttt{self.curr_img_id} = 0$$. Following this, $$\texttt{self.ground_truth_label}$$ is updated, and the corresponding label is sent to the receiving `Process` (i.e., to $$\texttt{OutSpkToCls}$$) in the same $$\texttt{self.time_step}=1$$. Finally, the encoding neurons voltage $$\texttt{self.v}$$ is reset to $$0$$ for fresh Rate-encoding of the new input image, and $$\texttt{self.curr_img_id}$$ is updated to the next value, i.e., $$1$$.

This marks the end of the $$\texttt{InpImgToSpk}$$'s $$\texttt{self.time_step}=1$$'s iteration.

* $$\texttt{self.time_step}=2$$:

At the beginning of $$\texttt{self.time_step}=2$$, the $$\texttt{run_spk()}$$ phase is once again called first. This time, the $$\texttt{if}$$ condition block is ignored, and $$\texttt{self.v}$$ is updated in accordance with the values of $$\texttt{self.inp_img}$$ (i.e., of the test-image at index $$\texttt{self.curr_img_id}=0$$). Subsequently, if $$\texttt{self.v}$$ crosses the threshold $$\texttt{self.vth}$$, then the spikes are generated and sent. In this same time-step, needless to say, $$\texttt{post_guard()}$$ returns $$\texttt{False}$$ and $$\texttt{run_post_mgmt()}$$ is _not_ invoked; thus, $$\texttt{self.inp_img}$$ remains unchanged.

* $$\texttt{self.time_step}=3$$:

Once again $$\texttt{run_spk()}$$ is the first phase to be exeucted with the same $$\texttt{self.inp_img}$$ values (as in the previous time-step), and the spikes are generated and sent; the $$\texttt{post_guard()}$$ phase evaluates to $$\texttt{False}$$ and $$\texttt{run_post_mgmt()}$$ is _not_ called.

This sort of execution continues all the way until the $$21^{\text{st}}$$ time-step. However, let's look at what's happening in $$\texttt{self.time_step}=20$$ and onwards:

* $$\texttt{self.time_step}=20$$:

The $$\texttt{run_spk()}$$ phase is executed first with the same image $$\texttt{self.inp_img}$$ at the index $$\texttt{self.curr_img_id}=0$$; $$\texttt{post_guard()}$$ returns $$\texttt{False}$$ and $$\texttt{run_post_mgmt()}$$ is _not_ called.

* $$\texttt{self.time_step}=21$$:

Now, note that $$\texttt{self.n_ts}=20$$ time-steps (i.e., the presentation time-steps of each image) has already passed, however, $$\texttt{self.inp_img}$$ is still the same _old_ image! Therefore, the $$\texttt{if}$$ block in $$\texttt{run_spk()}$$ is _necessary_ which resets the $$\texttt{self.inp_img}$$ and $$\texttt{self.v}$$ to all zeros; this is similar to the case when $$\texttt{self.time_step}=1$$. Subsequently, _no_ spikes are generated and sent to the receiving `Process` (i.e., to the `netx`-obtained network). However, after the $$\texttt{run_spk()}$$ phase is over, $$\texttt{post_guard()}$$ returns $$\texttt{True}$$ and the $$\texttt{run_post_mgmt()}$$ phase is executed, which updates $$\texttt{self.inp_img}$$ to the normalized pixel values of the test-image at index $$\texttt{self.curr_img_id}=1$$ (recollect that $$\texttt{self.curr_img_id}$$ was already updated to $$1$$ in $$\texttt{self.time_step}=1$$); $$\texttt{self.ground_truth_label}$$ is also updated accordingly and sent to the receiving `Process` (i.e., to $$\texttt{OutSpkToCls}$$), $$\texttt{self.v}$$ is reset to all zeros for fresh Rate-encoding of the new $$\texttt{self.inp_img}$$ image, and finally, $$\texttt{self.curr_img_id}$$ is updated to its next value, i.e., $$2$$.

* $$\texttt{self.time_step}=22$$:

The $$\texttt{run_spk()}$$ phase is called first, and it updates $$\texttt{self.v}$$ corresponding to the new $$\texttt{self.inp_img}$$ at index $$\texttt{self.curr_img_id}=1$$, thereby generating and sending the spikes if the threshold criterion is met; $$\texttt{post_guard()}$$ returns $$\texttt{False}$$ and $$\texttt{run_post_mgmt}$$ is _not_ called.

Henceforward, I believe, it should be easy to follow this repetition as long as the Lava network is run for. Also, before we miss the context, coming back to the $$\texttt{reset_offset}$$ in $$\texttt{netx.hdf5.Network()}$$, hopefully it should be clear now on why $$\texttt{reset_offset}$$ was set to $$1$$. This was done such that (in conjunction with the $$\texttt{reset_interval}=20$$) the `netx`-obtained network resets after $$21^{\text{st}}, 41^{\text{st}}, 61^{\text{st}}...$$ time-steps, because, as we saw above, the $$\texttt{self.inp_img}$$ gets reset (and assigned a new test-image) after $$21^{\text{st}}, 41^{\text{st}}, 61^{\text{st}}...$$ time-steps; thus, the `netx`-obtained network resets in _synchrony_ with the input test-image.

## $$\texttt{OutSpkToCls}$$ and $$\texttt{PyOutSpkToClsModel}$$ Classes
The `Process`: $$\texttt{OutSpkToCls}$$ and its corresponding `ProcessModel`: $$\texttt{PyOutSpkToClsModel}$$ implement the code to accept the **output** spikes from the `netx`-obtained network's **Output** interface and infer the predicted class by reporting the index which has the maximum accumulated spikes over the presentation time-steps (i.e., $$\texttt{self.n_ts}$$) of an image.

The following code in $$\texttt{utils.py}$$:
```python
@implements(proc=OutSpkToCls, protocol=LoihiProtocol)
@requires(CPU)
class PyOutSpkToClsModel(PyLoihiProcessModel):
```
associates the `Process`: $$\texttt{OutSpkToCls}$$ to its `ProcessModel`: $$\texttt{PyOutSpkToClsModel}$$ and instructs it to execute according to the `LoihiProtocol` on host `CPU`. As mentioend above, in the `LoihiProtocol`, the $$\texttt{run_spk()}$$ phase is executed first, followed by the $$\texttt{post_guard()}$$ phase, upon which the $$\texttt{run_post_mgmt()}$$ phase is conditioned.
### Components of $$\texttt{PyOutSpkToClsModel}$$
Following is the code in $$\texttt{run_spk()}$$ (in $$\texttt{utils.py}$$):
```python
def run_spk(self):
    spk_in = self.spikes_in.recv()
    self.spikes_accum = self.spikes_accum + spk_in
```
where $$\texttt{self.spikes_accum}$$ simply adds up the incoming spikes (every time-step) received in the $$\texttt{self.spikes_in}$$ port. As you might reckon, $$\texttt{self.spikes_accum}$$ should be reset after every $$\texttt{self.n_ts}=20$$ presentation time-steps. This is precisely what's done in the $$\texttt{run_post_mgmt()}$$ phase here. However, since the $$\texttt{run_post_mgmt()}$$ phase is guarded by the $$\texttt{post_guard()}$$ phase, we look into its code first:
```python
def post_guard(self):
    if self.time_step % self.n_ts == 0:
        return True

    return False
```
As can be seen above, whenever $$\texttt{self.time_step}$$ is a multiple of $$\texttt{self.n_ts}$$ (i.e., presentation time-steps of one image is over), $$\texttt{post_guard()}$$ returns $$\texttt{True}$$, setting the stage for $$\texttt{run_post_mgmt()}$$ to execute. Following is the code in $$\texttt{run_post_mgmt()}$$:
```python
def run_post_mgmt(self):
    true_label = self.label_in.recv()
    pred_label = np.argmax(self.spikes_accum)
    self.true_labels[self.curr_idx] = true_label[0]
    self.pred_labels[self.curr_idx] = pred_label
    self.curr_idx += 1
    self.spikes_accum = np.zeros_like(self.spikes_accum)
```
As can be seen above, when the $$\texttt{run_post_mgmt()}$$ phase is executed (i.e., after every $$\texttt{self.n_ts} = 20$$ time-steps), the $$\texttt{true_label}$$ is received from the sending `Process` (i.e., from $$\texttt{InpImgToSpk}$$) and the $$\texttt{pred_label}$$ is computed as the index having the maximum number of accumulated spikes. The respective label arrays are also populated at the current image index $$\texttt{self.curr_idx}$$, thereafter updating $$\texttt{self.curr_idx}$$ and resetting the $$\texttt{self.spikes_accum}$$ to store the output spikes corresponding to the next image (during the next $$\texttt{self.n_ts}$$ time-steps). We now look into the operation of all these three phases in tandem.
### Holistic Operation of $$\texttt{OutSpkToCls}$$
Let's begin by assuming $$\texttt{self.curr_idx}=0$$ at the start of running the $$\texttt{OutSpkToCls}$$ `Process`.

* $$\texttt{self.time_step}=1$$:

The $$\texttt{run_spk()}$$ phase is called first, which accepts the spikes (if produced from the `netx`-obtained network) corresponding to the input image at $$\texttt{self.curr_img_id}=0$$ and accumulates them in $$\texttt{self.spikes_accum}$$. Note that $$\texttt{post_guard()}$$ returns $$\texttt{False}$$ and $$\texttt{run_post_mgmt()}$$ is _not_ invoked.

* $$\texttt{self.time_step}=2$$:

The $$\texttt{run_spk()}$$ phase is called again - it receives the spikes (if produced from the `netx`-obtained network) and stores them in the $$\texttt{self.spikes_accum}$$. The $$\texttt{run_post_mgmt()}$$ phase is _not_ called as $$\texttt{post_guard()}$$ returns $$\texttt{False}$$.

This sort of processing continues for the test-image at $$\texttt{self.curr_img_id}=0$$ until the $$20^{\text{th}}$$ time-step begins, i.e.,

* $$\texttt{self.time_step}=20$$:

The $$\texttt{run_spk()}$$ phase is called first, it receives the spikes from the `netx`-obtained network and updates $$\texttt{self.spikes_accum}$$. This time the $$\texttt{post_guard()}$$ phase returns $$\texttt{True}$$ and the $$\texttt{run_post_mgmt()}$$ phase is called. It receives the ground truth label sent from the $$\texttt{InpImgToSpk}$$ `Process` and computes the predicted label for the current input image at $$\texttt{self.curr_img_id}=0$$ from $$\texttt{self.spikes_accum}$$ and populates the respective label arrays at the $$\texttt{self.curr_id}=0$$ index. It then updates $$\texttt{self.curr_id}$$ to its next value $$1$$ and resets the $$\texttt{self.spikes_accum}$$ to all zeros, thereby setting the stage for processing the next input image for the next $$\texttt{self.n_ts}=20$$ presentation time-steps.

* $$\texttt{self.time_step}=21$$:

Note that in this time-step, $$\texttt{self.inp_img}$$ and $$\texttt{self.v}$$ is reset to all zeros in the $$\texttt{run_spk()}$$ phase of the $$\texttt{InpImgToSpk}$$ `Process`. Also, $$\texttt{post_guard()}$$ (in $$\texttt{InpImgToSpk}$$) evaluates to $$\texttt{True}$$ and the $$\texttt{run_post_mgmt()}$$ phase is called in the $$\texttt{InpImgToSpk}$$ `Process`, thereby, updating $$\texttt{self.inp_img}$$ and $$\texttt{self.ground_truth_label}$$ to the next image and ground truth (at index $$\texttt{self.curr_img_id}=1$$) respectively; the updated $$\texttt{self.ground_truth_label}$$ is sent to this $$\texttt{OutSpkToCls}$$ `Process`.

Thus, in the $$\texttt{run_spk()}$$ phase of $$\texttt{OutSpkToCls}$$ `Process`, the spikes from the `netx`-obtained network corresponding to the _new_ test-image at $$\texttt{self.curr_img_id}=1$$ is processed. Although, $$\texttt{run_post_mgmt()}$$ is _not_ called because $$\texttt{post_guard()}$$ returns $$\texttt{False}$$.

Hopefully, it is now clear on how this repetition across all the time-steps progresses as long as the Lava network runs.

## Inferencing with $$\texttt{LavaDenseSNN}$$ on $$\textsf{Loihi-2}$$
We now finally bring our attention to deploying our trained SNN on $$\textsf{Loihi-2}$$ neuromorphic chip for inference. Following code in $$\texttt{snns.py}$$ builds the $$\texttt{run_config}$$ depending upon the backend $$\textsf{Loihi-Sim}$$ and $$\textsf{Loihi-Hw}$$ ($$\texttt{L2Sim}$$ and $$\texttt{L2Hw}$$ in code respectively):
```python
if backend == "L2Sim": # Run on the Loihi-2 Simulation Hardware on CPU.
    run_config = Loihi2SimCfg(
        select_tag="fixed_pt", # To select fixed point implementation.
        exception_proc_model_map={
            InpImgToSpk: PyInpImgToSpkModel,
            OutSpkToCls: PyOutSpkToClsModel,
            InputAdapter: PyInputAdapter,
            OutputAdapter: PyOutputAdapter
            }
        )
elif backend == "L2Hw": # Run on the Loihi-2 Physical Hardware on INRC.
    run_config = Loihi2HwCfg(
        select_sub_proc_model=True,
        exception_proc_model_map={
            InpImgToSpk: PyInpImgToSpkModel,
            OutSpkToCls: PyOutSpkToClsModel,
            InputAdapter: NxInputAdapter,
            OutputAdapter: NxOutputAdapter
            }
        )
```
The above $$\texttt{run_config}$$ is then used to $$\texttt{run()}$$ the $$\texttt{LavaDenseSNN}$$ on the appropriate backend with properly mapped `ProcessModel`s of the composing `Process`es. Now, the code below does the per-image inference (on the chosen backend):
```python
# Execute the trained network on each image indvidually.
for _ in range(self.num_test_imgs):
    self.img_to_spk.run(
        condition=RunSteps(num_steps=self.n_ts), run_cfg=run_config
    )
```
where $$\texttt{self.n_ts}$$ is simply the presentation time-steps of each test-image. Upon inferencing for all the $$10000$$ test-images on $$\textsf{Loihi-Sim}$$ (i.e., $$\texttt{backend == L2Sim}$$ on CPU), I am getting $$94.27\%$$ test accuracy. Note that the execution on $$\textsf{Loihi-Hw}$$ (i.e., $$\texttt{backend == L2Hw}$$ on INRC) requires the $$\texttt{reset_interval}$$ (in $$\texttt{netx.hdf5.Network()}$$ to be a power of $$2$$, hence, I have chosen it to be $$32$$ in my runs on INRC. And yes, don't forget to _stop_ running your `LavaDenseSNN` after obtaining the true and predicted classes!
```python
# Stop the run-time AFTER obtaining all the true and pred classes.
self.img_to_spk.stop()
```

# Closing Words

This was quite a long tutorial, by the virtue of being end-to-end. At the time of writing this, I couldn't find any comprehensive end-to-end tutorial on [Lava's website](https://lava-nc.org/); rather, the tutorials are/were broken into training via `slayer` and then evaluation via `netx` (both separate). Nonetheless, those Lava tutorials formed the essential building blocks of this end-to-end tutorial. Here, after a short introduction to Lava, I showed you how to first build your SNN using SLAYER, followed by its **Direct Training** on GPU, and then eventually porting it to either Loihi simulation-hardware or Loihi physical-hardware for inference. I also described how the different **phases** of a `ProcessModel` execute every time-step; this intricate understanding of `Process` execution is quite necessary to build complex Lava networks. I hope that this tutorial serves as a foundation to your next awesome Neuromorphic Computing/Lava project!

Feel free to comment below if you have any questions in understanding this tutorial or executing the code in [mnist-on-loihi](https://github.com/R-Gaurav/mnist-on-loihi) repository. And don't forget to leave a star on Github if this tutorial/repository helped you! Thank you for stopping by :).

---
