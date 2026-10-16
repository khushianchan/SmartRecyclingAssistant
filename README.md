

\# Smart Recycling Assistant ♻️



An AI-powered web application that classifies waste images into six categories and provides recycling and disposal guidance.



\## Features



\- Image-based waste classification using a TensorFlow Lite model

\- Six categories: cardboard, glass, metal, paper, plastic, and trash

\- Recommended waste treatment steps

\- SQLite-based prediction history

\- Dashboard with waste distribution chart and recent predictions



\## Technologies Used



\- Python

\- Flask

\- TensorFlow Lite

\- NumPy

\- Pillow

\- Matplotlib

\- SQLite

\- HTML and CSS



\## How to Run Locally



1\. Clone or download this repository.

2\. Create and activate a Python virtual environment.

3\. Install dependencies:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Ensure `waste\_classifier.tflite` is in the project directory.

5\. Run:



&#x20;  ```bash

&#x20;  python app.py

&#x20;  ```



6\. Open `http://127.0.0.1:5000` in your browser.



\## Model



The application uses a TensorFlow Lite model to classify uploaded waste images.



\## Limitations



Predictions may be incorrect. Model confidence is not the same as measured accuracy. Always follow local waste-management guidelines.



\## Future Enhancements



\- Improve classification accuracy with more training data

\- Add more waste categories

\- Deploy the application online

\- Improve recycling recommendations for local waste facilities



