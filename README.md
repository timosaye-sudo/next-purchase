# Next purchase prediction

Team project from the École des Ponts hackathon 2026, on a case proposed by eleven strategy. This repository is a simplified version of the code we used during the hackathon.

A sports clothing retailer wants to recommend the right products to each of its customers for marketing campaigns. For each customer, we predict the next product they will buy among almost 30,000 products, and we propose five of them. We first select 100 candidates per customer with simple counts, then a LightGBM model ranks them.

## How to run

The data is not included: it was given to the participants of the hackathon. Put `transactions.csv`, `clients.csv`, `products.csv` and `submission_template.csv` in `data/raw/`, then:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python reperes.py
python main.py
python soumission.py
```

`reperes.py` computes simple baselines, `main.py` trains and evaluates the model in about 15 seconds, and `soumission.py` writes the submission file.

Each file in `src/` can also be run on its own to check that it works, for example `python -m src.evaluation`.

## Results

Recall@5 on 3,000 test customers that were not used for training or tuning:

| Method | Recall@5 |
|---|---|
| Five random products | 0.02% |
| Five best sellers | 2.63% |
| Five best sellers in the customer's country | 3.00% |
| Last five products bought by the customer | 23.33% |
| Our model | 25.03% |

Just proposing the last five products bought already gives 23.3%, and the model adds about 1.7 points. On samples of 200 customers, the standard deviation is 3.2 points, so small differences between methods are not meaningful at that size.
