from itertools import pairwise

import numpy as np
import polars as pl

MIN_LIGNES = 4


def masque_dernier_achat(tx: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    numerote = (
        tx.with_row_index("ligne")
        .sort(["ClientID", "jour", "ligne"])
        .with_columns(pl.int_range(pl.len()).reverse().over("ClientID").alias("rang"))
    )
    # même filtre que l'organisateur : au moins 4 lignes d'achat
    eligibles = (
        tx.group_by("ClientID").len().filter(pl.col("len") >= MIN_LIGNES)["ClientID"]
    )
    a_masquer = numerote.filter(
        (pl.col("rang") == 0) & pl.col("ClientID").is_in(eligibles.to_list())
    )
    cible = dict(zip(a_masquer["ClientID"], a_masquer["ProductID"]))

    visible = numerote.join(a_masquer.select("ligne"), on="ligne", how="anti").drop(
        "rang", "ligne"
    )
    return visible, cible


def decoupe(liste_clients: list[str], tailles: dict, graine: int = 0) -> dict:
    melange = np.random.default_rng(graine).permutation(np.array(liste_clients))
    groupes, debut = {}, 0
    for nom, n in tailles.items():
        groupes[nom] = melange[debut : debut + n].tolist()
        debut += n
    return groupes


def recall_at_k(
    scores: np.ndarray, y: np.ndarray, groupes: np.ndarray, k: int = 5
) -> tuple[float, np.ndarray]:
    bornes = np.concatenate([[0], np.cumsum(groupes)])
    succes = np.zeros(len(groupes))

    for i, (a, b) in enumerate(pairwise(bornes)):
        if b <= a:
            continue
        if b - a <= k:
            indices = np.arange(a, b)
        else:
            indices = np.argpartition(-scores[a:b], k)[:k] + a
        succes[i] = max(y[indices])  # 1 si au moins un y==1, 0 sinon

    return succes.mean(), succes


if __name__ == "__main__":
    # deux clients : le bon produit est dans le top 5 du premier, pas du second
    scores = np.array([0.9, 0.1, 0.5, 0.4, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    print(recall_at_k(scores, y, np.array([6, 6]))[0])  # doit donner 0.5
