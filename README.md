### Local Deployment

1. Create a new conda environment

    ```conda create -n yolov8-object-conting-local-deployment python=3.8```

2. Activate this environment

    ```conda activate yolov8-object-conting-local-deployment```

3. Clone repo

    ```git clone https://github.com/grhaonan/yolov8-object-counting.git```

4. Install all dependencies

    ```cd yolov8-object-counting```

    ```pip install -r requirements.txt```

5. Start the Streamlit app and it should bring up the app at http://localhost:8501/

    ```streamlit run app.py```