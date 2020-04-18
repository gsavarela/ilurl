"""

    Python script to run full pipeline:

        1) jobs/train.py: train agent(s).
        2) analysis/train_plots.py: create training plots.
        3) jobs/rollouts.py: execute and evaluate different
                             Q-tables using rollouts.
        4) analysis/rollouts.py: create rollouts plots.

"""

from jobs.train import train_batch as train
from jobs.rollouts import rollout_batch as rollouts

from analysis.train_plots import main as train_plots
from analysis.rollouts import main as rollouts_plots


if __name__ == '__main__':

    # Train agent(s).
    experiment_root_path = train()

    # Create train plots.
    train_plots(experiment_root_path)

    # Execure rollouts.
    eval_path = rollouts(batch_dir=experiment_root_path)

    # Create rollouts plots.
    rollouts_plots(eval_path)

    print('EXPERIMENT FINALIZED')
    print('Experiment folder: {0}'.format(experiment_root_path))