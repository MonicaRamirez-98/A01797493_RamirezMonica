# Modelo cancelaciones de tarjeta de crédito – Experimentación con MLflow

Mónica María Ramírez Mejía (A01797493)

Regresión Logística para predecir la cancelación de clientes de tarjeta de crédito
(dataset [BankChurners](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)).

## Archivos
- `train.py`: script de entrenamiento parametrizado con registro en MLflow.
- `notebooks/01_EDA_BankChurners.ipynb`: exploración de datos (solo EDA).
- `requirements.txt`: dependencias con versiones fijadas (Python 3.13).

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
python train.py          # una corrida con valores por defecto
python train.py --help   # ver argumentos
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la regularización |
| `--penalty` | l2 | `l1` o `l2` |
| `--class-weight` | none | `balanced` compensa el desbalance |
| `--max-iter` | 1000 | Iteraciones máximas |
| `--seed` | 42 | Semilla aleatoria |
| `--run-name` | – | Nombre de la corrida |

### Corridas del experimento
```bash
python train.py --run-name experimento1
python train.py --C 0.01 --run-name experimento2
python train.py --C 100 --run-name experimento3
python train.py --penalty l1 --C 0.1 --run-name experimento4
python train.py --class-weight balanced --run-name experimento5
python train.py --class-weight balanced --penalty l1 --C 0.1 --run-name experimento6
```

| Corrida | C | penalty | class_weight | Qué cambia respecto a experimento1 |
|---|---|---|---|---|
| experimento1 | 1.0 | l2 | none | Nada: valores por defecto (mismo modelo del notebook) |
| experimento2 | 0.01 | l2 | none | Regularización más fuerte (C menor) |
| experimento3 | 100 | l2 | none | Regularización más débil (C mayor) |
| experimento4 | 0.1 | l1 | none | Regularización L1 en lugar de L2, con C = 0.1 |
| experimento5 | 1.0 | l2 | balanced | Pesos de clase balanceados para compensar el desbalance |
| experimento6 | 0.1 | l1 | balanced | Combina pesos balanceados con regularización L1 y C = 0.1 |

## MLflow
Cada corrida registra parámetros, métricas (accuracy, precision, recall, f1),
la matriz de confusión como artefacto y el modelo entrenado, en el experimento
**Credit Card Churn - Regresion Logistica**.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Abrir http://127.0.0.1:5000, seleccionar las corridas y pulsar **Compare**.

## Reproducibilidad
La semilla (`--seed`, por defecto 42) se aplica a `random`, `numpy`, a la partición
de datos y al modelo. Dos ejecuciones con los mismos argumentos dan las mismas métricas.