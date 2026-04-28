import romtools.workflows

from romtools.hpc.logger import Logger
from romtools.hpc.dispatcher import Dispatcher

from romtools.hpc.basic.BasicModel import BasicModel
from romtools.hpc.basic.BasicParameterSpace import BasicParameterSpace

if __name__ == '__main__':

    # This will be created both locally and off of remote_root
    output_dir_name = "samples_01"

    logger = Logger()
    with Dispatcher(
        logger,
        sampling_directory=output_dir_name,
    ) as dispatcher:

        model = BasicModel(dispatcher)
        params = BasicParameterSpace()
        num_samples = 5

        romtools.workflows.run_sampling(
            model = model,
            parameter_space = params,
            absolute_sampling_directory = output_dir_name,
            evaluation_concurrency = 5,
            number_of_samples = num_samples,
            random_seed = 1,
            dry_run = False,
            overwrite = True
        )
