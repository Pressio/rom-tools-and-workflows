import romtools.workflows

from romtools.hpc.remote_dispatcher import RemoteDispatcher

from romtools.hpc.example.ExampleModel import ExampleModel
from romtools.hpc.example.ExampleParameterSpace import ExampleParameterSpace

if __name__ == '__main__':

    # This will be created both locally and off of remote_root
    sampling_dir = "sample_all_collect"

    with RemoteDispatcher(sampling_dir) as dispatcher:

        model = ExampleModel()
        params = ExampleParameterSpace()
        num_samples = 1

        romtools.workflows.run_sampling(
            model = model,
            parameter_space = params,
            absolute_sampling_directory = sampling_dir,
            evaluation_concurrency = 1,
            number_of_samples = num_samples,
            random_seed = 1,
            dry_run = False,
            overwrite = True,
            dispatcher = dispatcher,
        )
