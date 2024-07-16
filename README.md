<!-- <img src="./figures/GemmaAIO-main-image.webp" alt="main-image"/> -->


## Main workflow
<!-- 
### Training the citation annotate model

```bash
sh train_citation.sh

``` -->

### Installation
To install this application, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/BachNgoH/AIO_Research_Assistant.git
cd AIO_Research_Assistant
```

**2. (Optional) Create and activate a virtual environment:**
- For Unix/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

- For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

### Data Ingestion

**Download data**
- Download the arxiv dataset [here](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) and put it in `./data` folder

- Download AIO document data:
```bash
cd data
git clone https://github.com/BachNgoH/AIO_Documents.git
```

**Ingest paper data**
```bash
python src/paper_ingest.py
```

**Ingest AIO Documents data**
```bash
python src/document_ingest.py
```

### Start application

```bash
# backend
uvicorn app:app --reload
# UI
streamlit run streamlit_ui.py

```

## Citation Extraction Workflow [WIP]

Comming soon
<img src="./figures/Graph-Paper-Search.jpg" alt="citation" width=800/>
<!-- 
**Generate citation data with**
```bash
sh gen_data.sh
``` -->


## Upcoming Checklist
- [ ] Update daily paper
- [ ] Graph search query engine
- [ ] UI
- [ ] Paper/Blog summary mode 
