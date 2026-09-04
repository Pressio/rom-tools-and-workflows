import romtools.workflows

from examples.dispatch_skill_example.AdrModel import Adr1dModel
from examples.dispatch_skill_example.AdrParameterSpace import AdrParameterSpace

if __name__ == '__main__':

    output_dir_name = "adr_sampling"

    model = Adr1dModel()
    params = AdrParameterSpace()

    romtools.workflows.run_sampling(
        model=model,
        parameter_space=params,
        absolute_sampling_directory=output_dir_name,
        evaluation_concurrency=8,
        number_of_samples=16,
        random_seed=1,
        dry_run=False,
        overwrite=True,
    )
