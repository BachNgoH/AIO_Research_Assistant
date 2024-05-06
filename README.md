# AIO Research Assistant


[notebook link](https://www.kaggle.com/code/bachngoh/gemma-data-assistant-w-llama-index-graph)

<img src="./figures/GemmaAIO-main-image.webp" alt="main-image"/>


## Main workflow

<img src="./figures/RAG-overview.jpg" alt="pipeline" width=800/>


### Training the citation annotate model

```bash
sh train_citation.sh

```

### Generate data

**Ingest paper data**
```bash
cd ./src
python ingest.py
```

**Generate citation data with**
```bash
sh gen_data.sh
```


## Citation Extraction Workflow

<img src="./figures/Graph-Paper-Search.jpg" alt="citation" width=800/>
