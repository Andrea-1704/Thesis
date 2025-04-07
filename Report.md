# What is this about?
We are going to add here each of the most importnat attempts we have done so far, in order to be sure to never miss out any of them and do it again!

# RandomForest_FE_baseline
Contains a trivial implementation of the position prediction for the f1 dataset provided by relbench.

Results:
Qui abbiamo un piccolo problema perché noto che dopo un po' di epoche il training mae rimane costante a crica 3.5 e validation mae a 3.10.

Per tutti gli esperimenti iniziali (quelli che mi portano a 3.10) ho usato solo due layers, quindi pensavo che il problema potesse essere quello. Dunque ho provato ad incrementare il numero di livelli a 5 ma peggioro il validation mae (3.17). [tutto considerando 100 epoche di training che probabilmente sono anche troppe].

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

Inoltre fin'ora la funzione di aggregazione è sempre stata la somma.

Proviamo adesso ad usare 3 livelli di GNN e aumentare questo lr

# train_model_baseline_f1

# train_GAT

# Graphormer

Results:

Attempts: