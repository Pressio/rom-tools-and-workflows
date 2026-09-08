import romtools.workflows

from romtools.hpc.logger import Logger

from romtools.hpc.example.ExampleModelNoConn import ExampleModelNoConn
from romtools.hpc.example.ExampleParameterSpace import ExampleParameterSpace

if __name__ == '__main__':

    # This will be created both locally and off of remote_root
    output_dir_name = "sample_00"

    logger = Logger()
    model = ExampleModelNoConn()
    params = ExampleParameterSpace()
    num_samples = 1

    romtools.workflows.run_sampling(
        model = model,
        parameter_space = params,
        absolute_sampling_directory = output_dir_name,
        evaluation_concurrency = 1,
        number_of_samples = num_samples,
        random_seed = 1,
        dry_run = False,
        overwrite = True,
    )
