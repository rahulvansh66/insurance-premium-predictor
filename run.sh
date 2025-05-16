#!/bin/bash
# chmod +x run.sh
# ./run.sh
echo "Starting Flask API..."
python app/api.py &

sleep 5

echo "Starting Streamlit App..."
streamlit run app/streamlit_app.py
