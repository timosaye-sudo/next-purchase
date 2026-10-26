import numpy as np
import polars as pl

from src import donnees, evaluation

TAILLES = {"train": 5000, "valid": 2000, "test": 3000}
K = 5


def main() -> None:
    tx, cl = donnees.transactions(), donnees.clients()
    visible, cible = evaluation.masque_dernier_achat(tx)
    test = evaluation.decoupe(sorted(cible), TAILLES, 0)["test"]

    # cinq produits au hasard
    n_produits = visible["ProductID"].n_unique()

    # les cinq produits les plus vendus, les mêmes pour tout le monde
    top = (
        visible.group_by("ProductID")
        .len()
        .sort(["len", "ProductID"], descending=[True, False])
        .head(K)["ProductID"]
        .to_list()
    )
    r_global = np.mean([cible[c] in top for c in test])

    # les cinq plus vendus dans son pays
    pays = dict(zip(cl["ClientID"], cl["ClientCountry"]))
    g = (
        visible.join(cl, on="ClientID", how="left")
        .group_by("ClientCountry", "ProductID")
        .len()
        .sort(["len", "ProductID"], descending=[True, False])
        .group_by("ClientCountry", maintain_order=True)
        .head(K)
    )
    top_pays = {
        cle[0]: s["ProductID"].to_list() for cle, s in g.group_by("ClientCountry")
    }
    r_pays = np.mean([cible[c] in top_pays.get(pays.get(c), []) for c in test])

    # ses cinq derniers produits achetés
    h = (
        visible.filter(pl.col("ClientID").is_in(test))
        .group_by("ClientID", "ProductID")
        .agg(pl.col("jour").max().alias("dernier"))
    )
    recents = {}
    for c, p, d in zip(h["ClientID"], h["ProductID"], h["dernier"]):
        recents.setdefault(c, []).append((d, p))
    r_rachat = np.mean(
        [
            cible[c] in [p for _, p in sorted(recents.get(c, []), reverse=True)[:K]]
            for c in test
        ]
    )

    print(f"hasard : {K / n_produits:.2%}")
    print(f"plus vendus du catalogue : {r_global:.2%}")
    print(f"plus vendus dans son pays : {r_pays:.2%}")
    print(f"cinq derniers achats : {r_rachat:.2%}")


if __name__ == "__main__":
    main()
