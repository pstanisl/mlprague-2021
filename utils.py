from scipy.stats import beta, norm
from typing import Any, Dict, Text

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Apply the default theme
sns.set_theme()


MOVIELENS_NUM_USERS = 943
MOVIELENS_NUM_MOVIES = 1682


def trajectory_for_bandit(initial_step, action_step, final_step):
  import tensorflow as tf
  from tf_agents.trajectories import trajectory

  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))


def load_movielens_data(data_file):
  """Loads the movielens data and returns the ratings matrix."""
  ratings_matrix = np.zeros([MOVIELENS_NUM_USERS, MOVIELENS_NUM_MOVIES])
  with open(data_file, 'r') as infile:
    # The file is a csv with rows containing:
    # user id | item id | rating | timestamp
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
      user_id, item_id, rating, _ = row
      ratings_matrix[int(user_id) - 1, int(item_id) - 1] = float(rating)
  return ratings_matrix


def create_accuracies(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df['price'] = 1
  df = pd.pivot_table(df, index='trial', columns=['action'], aggfunc=np.sum).price
  df['sum'] = df.sum(axis=1)
  df = df.loc[:,:].div(df["sum"], axis=0)
  df.columns = [f'action_{action}' for action in df.columns]

  return df.reset_index().fillna(0)


def plot_accuracy(df: pd.DataFrame, params: Dict[Text, Any]):
 plt.figure(figsize=(10, 7))

 for epsilon in df.epsilon.unique():
   df_acc = create_accuracies(df[df.epsilon == epsilon])

   plt.plot(
       df_acc.trial,
       df_acc[f'action_{params["best_action"]}'],
       label=f'action={params["best_action"]} (epsilon={epsilon})'
   )

 plt.xlim(0)
 plt.ylim(0, 1)
 plt.xlabel('Number of trials')
 plt.ylabel('Probability of Selecting Best Arm')
 plt.title(f'Accuracy of the best actions - {params["algorithm"]} Algorithm')
 plt.legend()
 plt.show()


def plot_actions(df: pd.DataFrame, params: Dict[Text, Any]):
  df_acc = create_accuracies(df)

  plt.figure(figsize=(10, 7))

  for action in sorted(df.action.unique()):
      plt.plot(
          df_acc.trial,
          df_acc[f'action_{action}'],
          label=f'action={action}'
      )

  plt.xlim(0)
  plt.xlabel('Number of Trials')
  plt.ylabel('Probability of Selecting Each Action')
  plt.legend()
  plt.title(f'Arm Selection Rate of the {params["algorithm"]} Algorithm')
  plt.show()


def plot_cumsum(df: pd.DataFrame, params: Dict[Text, Any], show_actions=False):
  plt.figure(figsize=(10, 7))

  df = df.copy()

  if show_actions:
    df_actions = df.groupby(['trial', 'action']).mean().reset_index()

    for action in sorted(df.action.unique()):
      plt.plot(
          df_actions[df_actions['action']==action].trial,
          df_actions[df_actions['action']==action].action_cum_sum,
          label=f'action={int(action)} cumulative reward'
      )

  df_all = df.groupby('trial')['cum_sum'].mean().reset_index()

  plt.plot(
    df_all.trial,
    df_all.cum_sum,
    linestyle='--',
    label='avg. agent cumulative reward'
  )

  plt.xlim(0)
  plt.xlabel('Number of trials')
  plt.ylabel('Cumulative Reward')
  plt.legend()
  plt.title(f'Cumulative Reward of the {params["algorithm"]} Algorithm')
  plt.show()


def plot_pdf(params: Dict[Text, Any], type: int = 0):
  plt.figure(figsize=(10, 7))

  if type == 0:
    x = np.linspace(
        beta.ppf(0.01, params['alpha'], params['beta']),
        beta.ppf(0.99, params['alpha'], params['beta']),
        100
    )
    lines = plt.plot(
        x, beta.pdf(x, params['alpha'], params['beta']), '-',
        lw=1, alpha=0.6, label='beta pdf')
  elif type == 1:
    x = np.linspace(
        norm.ppf(0.01, loc=params['mu'], scale=1/params['tau']),
        norm.ppf(0.99, loc=params['mu'], scale=1/params['tau']),
        100
    )
    lines = plt.plot(
        x, norm.pdf(x, loc=params['mu'], scale=1/params['tau']), '-',
        lw=1, alpha=0.6, label='norm pdf')

  #lines = ax.plot(x,y.T[:,:])
  plt.legend(lines, [f'action={j}' for j in range(len(lines))])
  plt.xlabel('x')
  plt.ylabel('PDF')
  plt.title(f'Probability density function - {params["dist"]}')
  plt.show()


def plot_regret(values, params: Dict[Text, Any]):
  plt.figure(figsize=(10, 7))

  plt.plot(values, label='regret')

  plt.xlim(0)
  plt.ylim(0)
  plt.title(f'Regret of {params["algorithm"]} on MovieLens environment')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Average Regret')
  plt.show()
