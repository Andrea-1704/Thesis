Best Val metrics: {'r2': 0.3251897151348895, 'mae': 3.0208979767166775, 'rmse': np.float64(3.808387545128338)}
Best test metrics: {'r2': 0.012922247923104924, 'mae': 4.250098149797372, 'rmse': np.float64(5.176611079479932)}

```

```

Best Val metrics: {'r2': 0.3251897151348895, 'mae': 3.0208979767166775, 'rmse': np.float64(3.808387545128338)}
Best test metrics: {'r2': 0.012922247923104924, 'mae': 4.250098149797372, 'rmse': np.float64(5.176611079479932)}

Best Val metrics: {'r2': 0.3251897151348895, 'mae': 3.0208979767166775, 'rmse': np.float64(3.808387545128338)}
Best test metrics: {'r2': 0.012922247923104924, 'mae': 4.250098149797372, 'rmse': np.float64(5.176611079479932)}

# What is this about?

We are going to add here each of the most importnat attempts we have done so far, in order to be sure to never miss out any of them and do it again!

# RandomForest_FE_baseline

Contains a trivial implementation of the position prediction for the f1 dataset provided by relbench.

Results:

# train_model_baseline_f1

This file contains the same implementation they did in relbench using the class GraphSAGE see the paper "GraphSAGE (minibatch Relbench)" in this repo for more informations.

I changed some ascects of their implementation such as the embedder, but the important things remained the same (they implemented).

Qui abbiamo un piccolo problema perché noto che dopo un po' di epoche il training mae rimane costante a crica 3.5 e validation mae a 3.10.

Per tutti gli esperimenti iniziali (quelli che mi portano a 3.10) ho usato solo due layers, quindi pensavo che il problema potesse essere quello. Dunque ho provato ad incrementare il numero di livelli a 5 ma peggioro il validation mae (3.17). [tutto considerando 100 epoche di training che probabilmente sono anche troppe, successivamente ne useremo **30** di epoche].

Per tutti questi esperimenti ho sempre usato i seguenti parametri di SGD:
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

Usando invece:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
ottengo un MAE=3.61 (val). Nota che con il primo optimizer indicato invece ottenevamo 3.15!

Forse dovremmo provare ad abbassare ulteriormente il lr, proviamo per esempio:
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
questo lr è troppo basso e la convergenza è davvero molto lenta. Nonostante sia molto lenta scende comunque molto molto bene(non abbiamo sali e scendi) MAE=3.14!.

Proviamo allora con questa configurazione:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
ottenendo un risultato migliore pari a MAE=3.13

Inoltre fin'ora la funzione di aggregazione è sempre stata la **somma**.

Proviamo adesso ad usare 3 livelli di GNN e aumentare questo lr

At the end I decided to use cross validation with the following hyper parameters in a grid search approach:

```python
for lr in [0.01, 0.001, 0.0001, 0.00001]:
    for batch_size in [64, 256, 512]:
        for num_layers in [1, 2, 3]:
            for weight_decay in [0.0001, 0.001, 0.01]:
                model = Model(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    num_layers=num_layers,
                    channels=128,
                    out_channels=1,
                    aggr="sum",
                    norm="batch_norm",
                ).to(device)
                print(f"Training with lr={lr}, batch_size={batch_size}, num_layers={num_layers}, weight_decay={weight_decay}")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                training_function(model, optimizer, epochs=30) # Set epochs to a smaller number for testing
```

Una cosa che manca da provare è vedere se usare differenti funzioni di aggregazione cambia il risultato. **Per qualche motivo a me non noto la funzione di aggregazione media resituisce dei risultati pessimi**: Best Val metrics: {'r2': -6.141562618877567, 'mae': 11.498565345146035, 'rmse': 12.389304125857201}.

I migliori parametri trovati fin'oro dopo il processo di cross validation sembrano essere questi:

```python
model = Model(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    num_layers=2,
                    channels=128,
                    out_channels=1,
                    aggr="mean",
                    norm="batch_norm",
                ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
```

Che ci portano ad un mae di circa 2.88.

#### Provare a mettere uno scheduler

# train_GAT

Questa implementazione parte dalla GNN baseline (Baseline_model.ipynb) ma utilizza una graph attention network invece della graphSAGE.

Cosa strana è che ancora non si riesce a migliorare le performance della baseline.

### Cross validation

Anche per questo modello abbiamo eseguito un processo di cross validation:

```python
for lr in [0.01, 0.001, 0.0001, 0.00001]:#0.001
    #for batch_size in [64, 256, 512]:
        for num_layers in [1, 2, 3]:#1
            #for weight_decay in [0.0001, 0.001, 0.01]:
                model = Model(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    num_layers=num_layers,
                    channels=128,
                    out_channels=1,
                    aggr="sum",
                    norm="batch_norm",
                ).to(device)
                print(f"Training with lr={lr}, num_layers={num_layers}")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
                training_function(model, optimizer, epochs=10) # Set epochs to a smaller number for testing
```

Dobbiamo ancora testare per gli ultimi due lr (0.0001, 0.00001) ma tra i primi due la combinazione di migliori iper parametri risulta essere ***lr=0.001*** e ***un solo layer***.

Per semplicità abbiamo fissato il valore di wd a quello precedentemente trovato.

Interessante notare che nonostante il paper che ha presentato le GAT (Graph Attention Network) usassero due livelli, la rete sembra funzionare molto meglio con un solo livello (questo risultat non solo è mostrato dalla cross validation, ma ho anche provato a fare un esperimento su 200 epoche in cui nel modello con un livello si arrivava ad una val mae di circa 2.79, mentre con due livelli a circa 3.65). Si potrebbe obiettare che il paper GAT trattava grafi omogenei.

Nota infatti che il nostro modello è un modello che segue l'implementazione HAN (Heterogeneous Graph Attention Networ) ovvero una GAT su un grafo eterogeneo. Possibili motivi per cui due livelli sono peggiori rispetto ad uno potrebbero essere:

1. **Più livelli di rete** significa  **più parametri** , e se il tuo modello ha molti parametri rispetto alla quantità di dati o alla complessità del problema, potrebbe iniziare a "memorizzare" piuttosto che generalizzare. Questo porta a un **overfitting** dove il modello si adatta troppo ai dati di addestramento e non generalizza bene su dati di validazione/test.
2. Nel caso delle **GAT** (Graph Attention Networks), l'uso di **multiple heads di attenzione** può essere molto utile in modelli con un singolo livello, ma quando aumenti il numero di livelli (num_layers=2), **l'aggregazione delle heads** potrebbe non essere più tanto efficace
3. Potrebbe esserci anche un errore nella configurazione dell'addestramento (ad esempio, il learning rate potrebbe essere troppo elevato per il modello con 2 livelli). Se il modello con 2 livelli sta facendo  **gradienti troppo grandi o troppo piccoli** , potrebbe non convergere correttamente.Potrebbe esserci anche un errore nella configurazione dell'addestramento (ad esempio, il learning rate potrebbe essere troppo elevato per il modello con 2 livelli). Se il modello con 2 livelli sta facendo  **gradienti troppo grandi o troppo piccoli** , potrebbe non convergere correttamente.

### Baseline vs GAT

La GAT sembra funzionare leggermente meglio raggiungendo MAE=2.83, rispetto a 2.92 della baseline.

# Graphormer basic

Dopo 100 epoche si raggiunge circa 3.13 di MAE.

Abbiamo proceduto ad implementare la versione finale di graphormer in maniera incrementale, ovvero dapprima abbiamo introdotto un semplice layer di self attention seguendo lo standard definito dal paper "Attention is all you need" ma evitando il positional encoding e aggiungendo invece un edge encoding.

Successivamente abbiamo costruito un secondo modello che aggiunge al livello precedente un concetto di node centrality calcolato tramite degree centrality.

Sembra che aggiungere il node centrality comporti una leggera riduzione del MAE sia nel set di validation che in quello di test, passando da:

```python
Best Val metrics: {'r2': 0.32290520643661624, 'mae': 3.0237126924072655, 'rmse': np.float64(3.8148285727717464)}
Best test metrics: {'r2': 0.0675817477108216, 'mae': 4.065723488372669, 'rmse': np.float64(5.031242368902457)}
```

Secondo esperimento:

```python
Best Val metrics: {'r2': 0.3251897151348895, 'mae': 3.0208979767166775, 'rmse': np.float64(3.808387545128338)}
Best test metrics: {'r2': 0.012922247923104924, 'mae': 4.250098149797372, 'rmse': np.float64(5.176611079479932)}
```

Nel caso di node encoding tramite degree centrality, a:

```python
Best Val metrics: {'r2': 0.3852869230994814, 'mae': 2.757049588449971, 'rmse': np.float64(3.634850185741035)}
Best test metrics: {'r2': -0.001497579711262187, 'mae': 4.226886181496738, 'rmse': np.float64(5.214285515336282)}
```

Secondo esperimento:

```python
Best Val metrics: {'r2': 0.3822518993244465, 'mae': 2.761465978909112, 'rmse': np.float64(3.6438123127543136)}
Best test metrics: {'r2': -0.08744494411399595, 'mae': 4.415857593762247, 'rmse': np.float64(5.433422727636282)}
```

Nel caso in cui non si usi il node encoding. In entrambi i due casi il modello è stato trainato su un massimo di 200 round e gli esperimenti sono stati eseguiti più volte per essere abbastanza sicuri che non fosse casuale.
