from itertools import pairwise

import numpy as np

VARIABLES = [
    # les trois sources
    "score_rachat",  # nombre d'achats du produit par le client
    "score_covis",  # somme des co-visitations depuis ses achats
    "score_pays",  # ventes du produit dans son pays
    "rang_covis",  # rang du produit dans la source co-visitation
    "nb_sources",  # combien des 3 sources le proposent
    # le client et le produit
    "deja_achete",  # 0 ou 1
    "jours_depuis",  # jours depuis son dernier achat de ce produit
    "nb_achats_produit",  # combien de fois il l'a acheté
    # le produit seul
    "prod_ventes",  # nombre de lignes de vente
    "prod_acheteurs",  # nombre d'acheteurs distincts
    # le client seul
    "cli_nb_lignes",  # taille de son historique
    "cli_nb_produits",  # nombre de produits distincts qu'il a achetés
]

JAMAIS = 9999.0  # jours_depuis quand il n'a jamais acheté le produit


def calcule(
    stats, gen: dict, histos: dict, liste_clients: list[str], jour_max: int
) -> np.ndarray:
    ic, ip, src = gen["client"], gen["produit"], gen["sources"]
    n = len(ic)
    X = np.zeros((n, len(VARIABLES)), dtype=np.float32)

    X[:, 0:3] = src

    bornes = np.concatenate(
        [[0], np.cumsum(np.bincount(ic, minlength=len(liste_clients)))]
    )

    for a, b in pairwise(bornes):
        if b <= a:
            continue
        X[a:b, 3] = np.argsort(-X[a:b, 1], kind="stable").argsort() + 1

    X[:, 4] = (src > 0).sum(axis=1)

    for k, (a, b) in enumerate(pairwise(bornes)):
        histo = histos.get(liste_clients[k], {})
        for r in range(a, b):
            dern, nb = histo.get(int(ip[r]), (None, 0))
            X[r, 5], X[r, 6], X[r, 7] = (
                (0, JAMAIS, 0) if dern is None else (1, jour_max - dern, nb)
            )

    X[:, 8] = np.log1p(stats.ventes[ip])
    X[:, 9] = np.log1p(stats.acheteurs[ip])

    for k, (a, b) in enumerate(pairwise(bornes)):
        histo = histos.get(liste_clients[k], {})
        X[a:b, 10] = np.log1p(sum(v[1] for v in histo.values()))
        X[a:b, 11] = np.log1p(len(histo))

    return X


def etiquettes(stats, gen: dict, cible: dict, liste_clients: list[str]) -> np.ndarray:
    vrai = np.array([stats.index.get(cible.get(c), -1) for c in liste_clients])
    return (gen["produit"] == vrai[gen["client"]]).astype(np.int8)


def groupes(gen: dict, n_clients: int) -> np.ndarray:
    n = np.bincount(gen["client"], minlength=n_clients)
    return n[n > 0]


if __name__ == "__main__":
    from . import candidats, donnees, evaluation

    tx = donnees.transactions()
    visible, cible = evaluation.masque_dernier_achat(tx)
    liste = sorted(cible)[:50]
    stats = candidats.construit(visible, donnees.clients())
    histos = candidats.historiques(visible, liste, stats.index)
    gen = candidats.genere(stats, histos, liste)
    X = calcule(stats, gen, histos, liste, int(visible["jour"].max()))
    print(X.shape)  # au plus 50 * 100 lignes et 12 colonnes
