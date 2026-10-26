import lightgbm as lgb
import numpy as np
import polars as pl

from src import candidats, donnees, modele, variables

K = 5
ITEMS = [f"item_{i}" for i in range(1, K + 1)]


def verifie(df: pl.DataFrame, gabarit: pl.DataFrame, catalogue: set) -> list[str]:
    controles = [
        ("nombre de lignes", df.height == gabarit.height),
        ("colonnes identiques", df.columns == gabarit.columns),
        (
            "ordre des ClientID",
            df["ClientID"].to_list() == gabarit["ClientID"].to_list(),
        ),
        ("colonne KEY intacte", df["KEY"].to_list() == gabarit["KEY"].to_list()),
        ("aucune valeur manquante", df.null_count().sum_horizontal().item() == 0),
        (
            "produits au catalogue",
            all(p in catalogue for c in ITEMS for p in df[c].to_list()),
        ),
        (
            "5 produits distincts par client",
            all(len({r[c] for c in ITEMS}) == K for r in df.iter_rows(named=True)),
        ),
    ]
    return [nom for nom, ok in controles if not ok]


def main() -> None:
    tx, cl, pr = donnees.transactions(), donnees.clients(), donnees.produits()
    gabarit = donnees.modele_soumission()
    a_predire = gabarit["ClientID"].to_list()

    stats = candidats.construit(tx, cl)
    histos = candidats.historiques(tx, a_predire, stats.index)
    gen = candidats.genere(stats, histos, a_predire)
    X = variables.calcule(stats, gen, histos, a_predire, int(tx["jour"].max()))

    m = lgb.Booster(model_file="modele.txt")
    scores = modele.predit(m, X)
    tops = modele.cinq_meilleurs(scores, gen, stats, len(a_predire), K)

    # si un client a moins de 5 candidats, on complète avec les meilleures ventes
    secours = list(stats.produits[np.argsort(-stats.ventes)[:20]])
    for i, liste in enumerate(tops):
        liste = list(dict.fromkeys(liste))
        j = 0
        while len(liste) < K:
            if secours[j] not in liste:
                liste.append(secours[j])
            j += 1
        tops[i] = liste[:K]

    df = gabarit.with_columns(
        [
            pl.col("ClientID")
            .replace_strict({c: tops[i][k] for i, c in enumerate(a_predire)})
            .alias(ITEMS[k])
            for k in range(K)
        ]
    ).select(["ClientID"] + ITEMS + ["KEY"])

    erreurs = verifie(df, gabarit, set(pr["ProductID"].to_list()))
    if erreurs:
        print("fichier non écrit, contrôles ratés :", ", ".join(erreurs))
    else:
        # ne pas ouvrir ce csv avec Excel : il tronque les identifiants à 15 chiffres
        df.write_csv("soumission.csv")
        print("soumission.csv écrit")


if __name__ == "__main__":
    main()
