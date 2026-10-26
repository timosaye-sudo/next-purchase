import numpy as np

from src import candidats, donnees, evaluation, modele, variables

TAILLES = {"train": 5000, "valid": 2000, "test": 3000}
GRAINE = 0


def prepare(stats, visible, liste, cible, jour_max):
    histos = candidats.historiques(visible, liste, stats.index)
    gen = candidats.genere(stats, histos, liste)
    X = variables.calcule(stats, gen, histos, liste, jour_max)
    y = variables.etiquettes(stats, gen, cible, liste)
    g = variables.groupes(gen, len(liste))
    return X, y, g, gen


def main() -> None:
    tx = donnees.transactions()
    cl = donnees.clients()

    # on cache le dernier achat de chaque client, c'est lui qu'il faut retrouver
    visible, cible = evaluation.masque_dernier_achat(tx)
    jour_max = int(visible["jour"].max())
    groupes_cli = evaluation.decoupe(sorted(cible), TAILLES, GRAINE)

    # les statistiques sont calculées seulement sur l'historique visible
    stats = candidats.construit(visible, cl)

    jeux = {}
    for nom, liste in groupes_cli.items():
        jeux[nom] = prepare(stats, visible, liste, cible, jour_max)

    X_tr, y_tr, g_tr, _ = jeux["train"]
    X_va, y_va, g_va, _ = jeux["valid"]
    X_te, y_te, g_te, _ = jeux["test"]
    m = modele.entraine(X_tr, y_tr, g_tr, X_va, y_va, g_va)

    r_va, _ = evaluation.recall_at_k(modele.predit(m, X_va), y_va, g_va)
    r_te, succes = evaluation.recall_at_k(modele.predit(m, X_te), y_te, g_te)

    # écart-type du recall sur des paquets de 200 clients, comme le classement final
    rng = np.random.default_rng(0)
    tirages = [succes[rng.integers(0, len(succes), 200)].mean() for _ in range(500)]

    m.save_model("modele.txt")
    print(f"Recall@5 validation : {r_va:.2%}, test : {r_te:.2%}")
    print(f"écart-type sur 200 clients : {np.std(tirages) * 100:.1f} points")


if __name__ == "__main__":
    main()
