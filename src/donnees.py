from pathlib import Path

import polars as pl

BRUT = Path(__file__).resolve().parents[1] / "data" / "raw"

# tout est lu comme du texte, sinon les identifiants à 17 chiffres sont abîmés
_TEXTE = {"infer_schema_length": 0}


def transactions() -> pl.DataFrame:
    return (
        pl.read_csv(BRUT / "transactions.csv", **_TEXTE)
        .select(
            pl.col("ClientID"),
            pl.col("ProductID"),
            # "2023-06-06 00:00:00+00:00" : on garde seulement la date
            pl.col("SaleTransactionDate").str.slice(0, 10).str.to_date().alias("date"),
        )
        # jour = nombre de jours depuis 1970, plus simple pour calculer les récences
        .with_columns(pl.col("date").dt.epoch("d").cast(pl.Int32).alias("jour"))
    )


def clients() -> pl.DataFrame:
    return pl.read_csv(BRUT / "clients.csv", **_TEXTE).select(
        "ClientID", "ClientCountry"
    )


def produits() -> pl.DataFrame:
    return pl.read_csv(BRUT / "products.csv", **_TEXTE).select("ProductID", "Category")


def modele_soumission() -> pl.DataFrame:
    return pl.read_csv(BRUT / "submission_template.csv", **_TEXTE)


if __name__ == "__main__":
    print(transactions().shape, clients().shape)  # environ 1.18 million de transactions
