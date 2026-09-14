# hpc

Execution layer for running Pressio-based ROM workflows on HPC clusters from
your local environment.

Everything goes through a single `Dispatcher` class, which manages the SSH
connection, file transfers, job submission, monitoring, and result collection
on your behalf. A `LocalDispatcher` implements the same interface without
sending work anywhere, so the same model runs either way.

Full documentation, including how to wire a dispatcher into your model and
workflow and how to configure it, lives at
[Remote Execution](https://pressio.github.io/rom-tools-and-workflows/remote_execution.html).

To see every configuration argument:

```sh
python -m romtools.hpc
```
