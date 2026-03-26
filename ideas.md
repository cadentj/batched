Problem

Currently NDIF runs user requests sequentially. A single job can block requests in queue, and p50 latency might increase under heavy usage periods.

What if we could batch requests? Some roadblocks: 
Job memory usage is dynamic.
The GPU can only execute one operation at a time.

Solution

Enabling concurrent GPU usage.

NVIDIA Multi-Process Service allows different kernels to share one scheduler and execute on SMs concurrently. With MPS, we can run jobs concurrently, with some caveats:
HBM is still a bottleneck for large tensor loads. If a user uploads a large tensor, their job might saturate HBM bandwidth and prevent other jobs from running. 
Bursty operations can cause traffic in L2 cache memory and block kernel launches.

Subproblem: 
What happens if a process OOMs? 
This should *never* given we sandbox correctly. Overallocating in Torch is just a program level error and won’t break the execution of other processes. On NDIF, we don’t allow custom kernels so we should never see invalid memory access issues / CUDA failures which break other jobs.

Capping the GPU usage of a job.

Torch provides `set_per_process_memory_fraction` methods on the `torch.mps` and `torch.cuda` packages to cap the memory usage of a process. 

Notably, memory only counts towards a process on tensor creation, so passing tensors between processes doesn’t count toward that process’s memory usage.

Also this doesn’t bound the HBM bandwidth for a job launch, so there can still be traffic.

Executing multiple jobs concurrently.

We can execute NNsight requests on separate processes and run model inference on another process. This isn’t too different from how NNsight currently works with model and worker threads.

Two subproblems: 
Passing tensors between processes is slow.
We can pass tensors between processes with Torch IPC. Rather than passing tensor data around, Torch IPC passes metadata between processes.
Passing tensors between processes cuts the autograd graph. 
Write a custom torch.autograd.Function which preserves the autograd graph between processes.

Batching user requests

Different jobs take a different amount of time at certain hookpoints. A single job could stall for minutes at a hookpoint.

Run user requests in parallel. If a user takes a while at a hookpoint, drop it from the batch and continue executing other interventions. Then pop new jobs from the queue for jobs which finished and run another forward pass. Once you arrive at the hookpoint which was dropped last iteration, add it back to the batch and continue.

The maximum delay added if a hookpoint is dropped is roughly the duration of a single forward pass which is basically negligible.

Two subproblems: 
User requests may be of drastically different sequence lengths. We’d waste a lot of compute on pad tokens.
Solution: We can implement basic triton kernels to do variable-length flattened sequence matmuls. These kernels execute separate matmuls per sequence, trading off slight overhead to save FLOPs.
Users may request multiple sequences at a time.
Solution: We cap the number of tokens in a batch. 

Tinker (link) is a different API from Thinky where users can fine tune adapters. Their API similarly executes in lockstep. Even though a smaller batch size is faster, a user’s request might take longer since Tinker pads multiple users’ jobs. They only charge users for the amount of compute they use rather than their job duration.

Existing Issues 

One thing I’m not really sure how to resolve yet is memory fragmentation between requests. Tensors are allocated in contiguous memory blocks. Certain requests might block memory and prevent new batches from being processed. Inference solves this by reserving memory upfront and allocating pages, but this doesn’t map cleanly onto NNsight jobs.

We also just couldn’t support: 
Source code editing.
We’d expose different model hookpoints than what huggingface transformers do if we add these custom kernels. That might break user code if they expect to be able to just set remote=True.
