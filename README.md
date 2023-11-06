# Flight Price Prediction

This project, part of the [#mlzoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/) course, focuses on predicting flight prices for routes connecting India's top six metro cities. It leverages a dataset containing various features such as airline, duration, class, etc., to create accurate price predictions. The goal is to assist travelers in making informed decisions about their flights.

To achieve this, three machine learning models - Linear Regression, Decision Tree, and Random Forest - have been trained and evaluated. The model with the best performance will be used for making predictions.

## Table of Contents
- [Data](#data)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Containerization](#containerization)

## Data

### Dataset

The flight booking dataset was obtained from the "Ease My Trip" website. 'Easemytrip' is an internet platform for booking flight tickets, and hence a platform that potential passengers use to buy tickets.

Data was collected for 50 days, from February 11th to March 31st, 2022. The data source was secondary data and was collected from the Ease My Trip website.

Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities. There are 60030 datapoints and 11 features in the dataset.

You can find the dataset [here](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

### Features

The various features of the cleaned dataset are explained below:
1) Airline: The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.
2) Flight: Flight stores information regarding the plane's flight code. It is a categorical feature.
3) Source City: City from which the flight takes off. It is a categorical feature having 6 unique cities.
4) Departure Time: This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and has 6 unique time labels.
5) Stops: A categorical feature with 3 distinct values that stores the number of stops between the source and destination cities.
6) Arrival Time: This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.
7) Destination City: City where the flight will land. It is a categorical feature having 6 unique cities.
8) Class: A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.
9) Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.
10) Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.
11) Price: Target variable stores information of the ticket price.

## Getting Started

Due to the binary model file 'model_rf.bin' exceeding 100MB, this repository utilizes [Git LFS (Large File Storage)](https://git-lfs.com/) for versioning. Make sure you have Git LFS installed on your system before cloning the repository.
Or you can generate a new model file with `train.py`.

## Prerequisites

To run this project, you will need to have the following installed:

- Python 3.10
- Pipenv (for managing the virtual environment and dependencies)
- Docker

## Installation

To get started, follow these steps:

1. Clone the repository.
2. Open a terminal and navigate to the project directory.
3. Run the following command to install the dependencies using Pipenv:

```bash
pipenv install
```
4. Install the dev-packages:

```bash
pipenv install -d
```

## Usage

### Notebook

In the `notebook.ipynb` file you'll find:
  - Data preparation and cleaning process
  - EDA and analyze feature importance
  - Model Selection and Parameter Tuning


### Train model

To train the Random Forest model and save it along with the Dictionary Vectorizer, use the `train.py` file provided in this repository. Running `train.py` will generate the following binary files:

- `model_rf.bin`: The trained Random Forest model.
- `dv.bin`: The Dictionary Vectorizer used for feature extraction.

#### Running the Training Script

```bash
pipenv run python train.py
```
After running the script, you will find the model_rf.bin and dv.bin files in the same directory as train.py.

### Serving the Model

To serve the model, use the Flask library. The `predict.py` file creates a server on port 9696. When a POST request is sent to the path '/predict' with the input data in the request body, it returns the model's prediction.

#### Running the Server

```bash
pipenv run python predict.py
```
Once the server is up and running, you can send a POST request to http://localhost:9696/predict with the input data to get predictions from the model.

#### Example Usage

```bash
$ curl -X POST -H "Content-Type: application/json" -d '{
  "airline": "air_india",
  "flight": "ai-619",
  "source_city": "mumbai",
  "departure_time": "night",
  "stops": "one",
  "arrival_time": "afternoon",
  "destination_city": "delhi",
  "class": "business",
  "duration": 19,
  "days_left": 17
}' http://localhost:9696/predict
```
This cURL request sends a POST request to the server running at http://localhost:9696/predict with a JSON body containing flight information. The server returns a response of:
```bash
41101.0
```

## Containerization

To containerize this project, follow these steps:

1. Ensure you have Docker installed on your system. If not, you can [download it here](https://www.docker.com/get-started).

2. Open a terminal and navigate to the project directory.

3. Run the following command to build a Docker image for the project:

```bash
docker build -t price-prediction .
```

This command will use the Dockerfile provided in the repository to build an image named price-prediction based on the python:3.10-slim image.

4. Once the image is built, you can run it in a Docker container with the following command:

```bash
docker run -it -p 9696:9696 price-prediction:latest
```

This will start a container running the project on port 9696.