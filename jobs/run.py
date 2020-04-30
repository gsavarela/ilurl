"""

    Python script to run full pipeline:

        1) jobs/train.py: train agent(s).

        2) analysis/train_plots.py: create training plots.

        3) jobs/rollouts.py: execute and evaluate different
                             Q-tables using rollouts.

        4) analysis/rollouts.py: create rollouts plots.

        5) jobs/rollouts.py: execute python script in test mode
                             in order to assess final agent performance.

                             Generates *.xml files and
                             *.test.info.json file.

        6) [Convert .xml files to .csv]

        7) analysis/test_plots.py: Create plots with metrics
                             for the final agent.

"""
from pathlib import Path

from jobs.train import train_batch as train
from jobs.rollouts import rollout_batch as rollouts

from analysis.train_plots import main as train_plots
from analysis.rollouts import main as rollouts_plots
from analysis.test_plots import main as test_plots

from ilurl.loaders.xml2csv import main as xml2csv

if __name__ == '__main__':

    # 1) Train agent(s).
    experiment_root_path = train()

    # 2) Create train plots.
    train_plots(experiment_root_path)

    # 3) Execute rollouts.
    eval_path = rollouts(batch_dir=experiment_root_path)

    # 4) Create rollouts plots.
    rollouts_plots(eval_path)

    # 5) Execute rollouts with last saved Q-tables (test).
    rollouts(test=True, batch_dir=experiment_root_path)

    # 7) Convert .xml files to .csv files.
    for xml_path in Path(experiment_root_path).rglob('*.xml'):
        csv_path = str(xml_path).replace('xml', 'csv')
        args = [str(xml_path), '-o', csv_path]
        try:
            xml2csv(args)
            Path(xml_path).unlink()
        except Exception:
            raise

    # 8) Create plots with metrics plots for final agent.
    test_plots(experiment_root_path)

    print('\nEXPERIMENT FINALIZED')
    print('Experiment folder: {0}'.format(experiment_root_path))
