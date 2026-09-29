# NetworkProject

Questo progetto esplora diverse tecniche per trasformare tracce di traffico di rete in immagini, al fine di effettuare **anomaly detection** utilizzando Reti Neurali Convoluzionali (CNN). 

Il progetto valuta il **trade-off** tra:
* Qualità del **training** finale.
* **Overhead** computazionale richiesto.
* Perdita di informazioni dovuta alla codifica in immagine.
* Espansione dei dati (byte) necessari per rappresentare la traccia, con un'attenzione particolare per i dispositivi **constrained** come quelli in ambito IoT.

## Struttura del Progetto

Il **repository** esplora le seguenti tecniche di conversione in immagine:

* `/dataset/`: Contiene i file CSV di input, come `CICIDS2017_sample.csv`.
* `/deepinsight/`: Implementazione della conversione DeepInsight. Contiene gli script di preprocessing, fit e transform.
* `/deepInsight-3D/`: Variante 3D del metodo DeepInsight.
* `/gaf/` e `/igaf/`: Tecniche basate su Gramian Angular Field.

## Prerequisiti e Installazione

Per eseguire questo progetto, è necessario Python 3.x. I **third-party packages** (pacchetti di terze parti) richiesti sono gestiti tramite pip.

1. Clona il repository:
   ```bash
   git clone https://github.com/ThetaLogN/NetworkProject.git
   cd NetworkProject