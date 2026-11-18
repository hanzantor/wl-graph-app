# Graph Analyzer: Weisfeiler–Leman Engine

A web tool for exploring graph isomorphism with the **k-dimensional Weisfeiler–Leman (k-WL)** algorithm. Upload or paste graphs, run refinement for a chosen k, and see how colour classes evolve and whether two graphs can be told apart.

## Features

- Parses several edge-list formats: one edge per line, all edges on one line, or `(u,v)` pairs
- Runs k-WL refinement and compares graph signatures to test for non-isomorphism, with an explanation of where two graphs diverge
- Visualises every refinement iteration side by side, with nodes coloured by class (NetworkX + Matplotlib)
- Single-page front end served by a Flask API

## Stack

Python · Flask · NetworkX · Matplotlib · HTML/CSS/JS · Gunicorn (Procfile-ready for Heroku/Render)

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000. To deploy, use `gunicorn app:app`.

## Author

**Hanin Al-Naamani**, MSc Advanced Data Science & AI, University of Liverpool
