#!/bin/bash
pip install -r requirements.txt
streamlit run frontend/app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
