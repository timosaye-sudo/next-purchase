from itertools import pairwise

import lightgbm as lgb
import numpy as np

from .variables import VARIABLES

# lambdarank : le modèle apprend à classer entre eux les candidats de chaque client
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 0,
    # mêmes résultats à chaque exécution
    "deterministic": True,
    "force_row_wise": True,
}


def entraine(X_tr, y_tr, g_tr, X_va, y_va, g_va, max_arbres: int = 600):
    d_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=VARIABLES)
    d_va = lgb.Dataset(X_va, label=y_va, group=g_va, reference=d_tr)
    return lgb.train(
        PARAMS,
        d_tr,
        num_boost_round=max_arbres,
        valid_sets=[d_va],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )


def predit(modele, X: np.ndarray) -> np.ndarray:
    return modele.predict(X, num_iteration=modele.best_iteration)


def cinq_meilleurs(scores, gen, stats, n_clients: int, k: int = 5) -> list:
    bornes = np.concatenate(
        [[0], np.cumsum(np.bincount(gen["client"], minlength=n_clients))]
    )
    sortie = []
    for a, b in pairwise(bornes):
        if b <= a:
            sortie.append([])
            continue
        top = np.argsort(-scores[a:b])[:k]
        sortie.append(list(stats.produits[gen["produit"][a:b][top]]))
    return sortie


if __name__ == "__main__":
    # données au hasard : on vérifie seulement que l'entraînement et la prédiction tournent
    rng = np.random.default_rng(0)
    X = rng.random((400, len(VARIABLES))).astype(np.float32)
    y = (rng.random(400) < 0.1).astype(np.int8)
    g = np.full(20, 20)
    m = entraine(X, y, g, X, y, g, max_arbres=5)
    print(predit(m, X).shape)  # doit donner (400,)
