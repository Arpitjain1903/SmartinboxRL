#!/bin/bash

# Start Streamlit on port 8501
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Start FastAPI on port 7860 (Hugging Face's public port)
echo "Starting OpenEnv API on port 7860..."
python main.py
