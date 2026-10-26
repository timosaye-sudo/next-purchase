from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

N_CANDIDATS = 100
N_SOURCES = 3
MAX_PRODUITS_COVIS = 50  # pour éviter d'associer une covisitation à tout

# calculés avec un modèle plus complexe et une optimisation des hyperparamètres
ALPHA, BETA = 0.75, 0.20


@dataclass
class Stats:
    produits: np.ndarray  # identifiants, dans l'ordre des indices
    index: dict  # identifiant -> indice
    ventes: np.ndarray  # nombre de lignes par produit
    acheteurs: np.ndarray  # nombre d'acheteurs distincts par produit
    covis: sp.csr_matrix  # covis[i, j] : score de i puis j
    top_pays: dict  # pays -> (indices produits, scores), déjà trié
    pays: dict  # client -> pays


def construit(visible: pl.DataFrame, clients_df: pl.DataFrame) -> Stats:
    produits = np.array(sorted(visible["ProductID"].unique().to_list()))
    index = {p: i for i, p in enumerate(produits)}
    ip = visible["ProductID"].replace_strict(index).to_numpy()

    ventes = np.bincount(ip, minlength=len(produits)).astype(np.float64)
    acheteurs = (
        visible.group_by("ProductID")
        .agg(pl.col("ClientID").n_unique().alias("n"))
        .with_columns(pl.col("ProductID").replace_strict(index).alias("i"))
    )
    acheteurs_arr = np.zeros(len(produits))
    acheteurs_arr[acheteurs["i"].to_numpy()] = acheteurs["n"].to_numpy()

    covis = _covisitation(visible, index, ventes)
    top_pays = _top_par_pays(visible, clients_df, index)
    pays = dict(zip(clients_df["ClientID"], clients_df["ClientCountry"]))
    return Stats(produits, index, ventes, acheteurs_arr, covis, top_pays, pays)


def _covisitation(
    visible: pl.DataFrame, index: dict, ventes: np.ndarray
) -> sp.csr_matrix:
    paires = visible.select("ClientID", "ProductID").unique()
    limite = (
        paires.group_by("ClientID")
        .len()
        .filter(pl.col("len") <= MAX_PRODUITS_COVIS)["ClientID"]
    )
    paires = paires.filter(pl.col("ClientID").is_in(limite.to_list()))

    cl = paires["ClientID"].unique().to_list()
    ic = paires["ClientID"].replace_strict({c: i for i, c in enumerate(cl)}).to_numpy()
    ip = paires["ProductID"].replace_strict(index).to_numpy()

    # M[client, produit] = 1 s'il l'a acheté,
    # donc (M.T @ M)[i, j] = nombre de clients qui ont acheté i et j
    M = sp.csr_matrix(
        (np.ones(len(ic), np.float32), (ic, ip)), shape=(len(cl), len(ventes))
    )
    C = (M.T @ M).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    # on divise par les popularités pour ne pas favoriser les produits très vendus
    C = C.tocoo()
    C.data = C.data / (ventes[C.row] ** ALPHA * ventes[C.col] ** BETA)
    return C.tocsr()


def _top_par_pays(
    visible: pl.DataFrame, clients_df: pl.DataFrame, index: dict, n: int = 60
) -> dict:
    j = visible.join(clients_df, on="ClientID", how="left")
    g = (
        j.group_by("ClientCountry", "ProductID")
        .len()
        .sort(["len", "ProductID"], descending=[True, False])
        .group_by("ClientCountry", maintain_order=True)
        .head(n)
    )
    out = {}
    for cle, sous in g.group_by("ClientCountry"):
        idx = sous["ProductID"].replace_strict(index).to_numpy()
        out[cle[0]] = (idx, sous["len"].to_numpy().astype(np.float64))
    return out


# pour chaque client : indice produit -> (dernier jour, nombre d'achats)
def historiques(visible: pl.DataFrame, liste_clients: list[str], index: dict) -> dict:
    sous = visible.filter(pl.col("ClientID").is_in(liste_clients))
    # trié pour que l'ordre, et donc les résultats, soient toujours les mêmes
    g = (
        sous.group_by("ClientID", "ProductID")
        .agg(pl.col("jour").max().alias("dernier"), pl.len().alias("n"))
        .sort("ClientID", "ProductID")
    )
    h = {c: {} for c in liste_clients}
    for cid, pid, dern, n in zip(g["ClientID"], g["ProductID"], g["dernier"], g["n"]):
        i = index.get(pid)
        if i is not None:
            h[cid][i] = (dern, n)
    return h


def _ajoute(scores, i, fente, valeur):
    if i not in scores:
        scores[i] = [0.0] * N_SOURCES
    scores[i][fente] = max(scores[i][fente], valeur)


def genere(stats: Stats, histos: dict, liste_clients: list[str]) -> dict:
    l_cli, l_prod, l_src = [], [], []

    for k, cid in enumerate(liste_clients):
        histo = histos.get(cid, {})
        scores = {}

        # source 1 : les produits déjà achetés par le client
        for i, (dern, n) in histo.items():
            _ajoute(scores, i, 0, float(n))

        # source 2 : la co-visitation depuis ses achats
        if histo:
            lignes = stats.covis[list(histo)]  # une ligne par produit acheté
            somme = np.asarray(lignes.sum(axis=0)).ravel()
            meilleurs = np.argpartition(-somme, min(80, len(somme) - 1))[:80]
            for i in meilleurs:
                if somme[i] > 0:
                    _ajoute(scores, int(i), 1, float(somme[i]))

        # source 3 : les meilleures ventes de son pays
        idx, val = stats.top_pays.get(stats.pays.get(cid), (np.array([], int), []))
        for i, v in zip(idx[:40], val[:40]):
            _ajoute(scores, int(i), 2, float(v))

        # d'abord les produits proposés par le plus de sources, puis la co-visitation
        items = sorted(
            scores.items(), key=lambda x: (-sum(v > 0 for v in x[1]), -x[1][1])
        )[:N_CANDIDATS]
        for i, s in items:
            l_cli.append(k)
            l_prod.append(i)
            l_src.append(s)

    return {
        "client": np.array(l_cli, np.int32),
        "produit": np.array(l_prod, np.int32),
        "sources": np.array(l_src, np.float32),
    }


if __name__ == "__main__":
    from . import donnees, evaluation

    tx = donnees.transactions()
    visible, cible = evaluation.masque_dernier_achat(tx)
    liste = sorted(cible)[:50]
    stats = construit(visible, donnees.clients())
    gen = genere(stats, historiques(visible, liste, stats.index), liste)
    print(np.bincount(gen["client"]).max())  # au plus 100 candidats par client
