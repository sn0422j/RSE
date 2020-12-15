from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rse import RSE

def plot_fig(df):
    sns.set_context("notebook")
    plt.Figure(dpi=150)
    sns.barplot(data=df, x="Model", y="MAE")
    plt.title("Simulation of RSE")
    plt.savefig("simulation.png")

def main(): 
    # S is Stimulus Featues, B is Neural Activity.  
    S, B = make_regression(n_samples=100, n_features=500, n_targets=500, random_state=0)
    print(f'the shape of S (Stimulus Featues): { S.shape } \nthe shape of B (Neural Activity): { B.shape }')
    cv_results = cross_validate(estimator=RSE(), X=S, y=B, scoring=make_scorer(mean_absolute_error))
    print('MAE score:', cv_results['test_score'])

    S1, B = make_regression(n_samples=100, n_features=500, n_targets=500, random_state=0)
    S2, _ = make_regression(n_samples=100, n_features=500, n_targets=500, random_state=1)
    S3, _ = make_regression(n_samples=100, n_features=500, n_targets=500, random_state=2)
    S1_score = cross_validate(estimator=RSE(), X=S1, y=B, scoring=make_scorer(mean_absolute_error))['test_score']
    S2_score = cross_validate(estimator=RSE(), X=S2, y=B, scoring=make_scorer(mean_absolute_error))['test_score']
    S3_score = cross_validate(estimator=RSE(), X=S3, y=B, scoring=make_scorer(mean_absolute_error))['test_score']
    score = [["S1", score] for score in S1_score] + [["S2", score] for score in S2_score] + [["S3", score] for score in S3_score]
    plot_fig(pd.DataFrame(score, columns=["Model", "MAE"]))

if __name__ == "__main__":
    main()
