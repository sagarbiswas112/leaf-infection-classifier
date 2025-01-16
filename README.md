# leaf-infection-classifier
This GitHub repository contains a comprehensive supervised machine learning project for accurately classifying plant diseases using Python. Leveraging the power of computer vision, we have employed CV2 (OpenCV), along with pandas, numpy, and matplotlib, for robust data preprocessing, exploration, and visualization.

Project Highlights:

Data Preprocessing: We prioritize data quality by performing thorough image processing and manipulation using CV2. Additionally, pandas and numpy enable seamless data handling, ensuring our model's efficiency.

Model Training: Our classifier achieves impressive results by employing Keras and TensorFlow with the powerful VGG19 architecture. This pre-trained model has been fine-tuned on a vast dataset of plant disease images, empowering it to recognize various diseases with remarkable accuracy.

Web Deployment: Experience the ease of plant disease identification through our intuitive web interface. Built with the Flask web framework, our deployment enables users to conveniently interact with the classifier, empowering farmers and botanists in diagnosing crop diseases.

Prerequisites:

Before running the project, ensure you have the following installed:

Keras
NumPy
TensorFlow
Matplotlib
Flask
We recommend using Conda to manage dependencies and create a virtual environment for seamless reproducibility.

Getting Started:

Clone this repository to your local machine.

Bash
Copy code
git clone https://github.com/sagarbiswas112/leaf-infection-classifier.git
Create a Conda environment using the provided environment file.

bash
Copy code
conda env create -f environment.yml
Activate the newly created Conda environment.

sql
Copy code
conda activate plant-disease-classifier
Run the Flask app to deploy the model on a local server.

Copy code
python app.py
Access the classifier through your web browser using the provided local server address.

Contributions:

Contributions to this project are encouraged! Whether you want to enhance model accuracy, improve the user interface, or expand the dataset, please submit your pull requests. Let's collaborate and create an impactful tool to combat plant diseases effectively.

License:

This project is licensed under the Apache License 2.0, allowing for flexible usage and distribution of the code. Please see the LICENSE file for more details.

Help us revolutionize plant disease detection and contribute to a greener, healthier environment! 🌿🍃🌱
