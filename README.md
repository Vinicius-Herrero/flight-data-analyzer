# Flight Data Analyzer

A Python project for exploratory data analysis (EDA) of flight data, including descriptive statistics, visualizations, and missing value handling.

## Project Structure

```
flight-data-analyzer/
├── database/              # Raw data files (downloaded from Google Drive)
│   ├── airlines.csv
│   ├── airports.csv
│   └── flights.csv
├── scripts/               # Analysis scripts
│   ├── download_data.py   # Download data from Google Drive
│   └── eda.py             # Exploratory data analysis
├── venv/                  # Virtual environment (not tracked)
├── visualizations/        # Visualizations
│   └── flight_eda.py      # Data visualization
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data from Google Drive

⚠️ **Important:** Data files are not included in the repository due to file size limits. Download them first:

```bash
python scripts/download_data.py
```

This will download `airlines.csv`, `airports.csv`, and `flights.csv` to the `database/` folder.

**Note:** The Google Drive folder must be accessible. If you don't have access, contact the repository owner.

## Usage

### Run Exploratory Data Analysis

Once data is downloaded, run the EDA:

```bash
python scripts/eda.py
```

This script will:
- Load and inspect three datasets (airlines, airports, flights)
- Generate descriptive statistics
- Analyze missing values
- Create visualizations (saves as `flight_eda_visualizations.png`)
- Provide recommendations for handling missing data

## Data Description

### Airlines
- IATA_CODE: Airline identifier
- AIRLINE: Airline name

### Airports
- IATA_CODE: Airport identifier
- AIRPORT: Airport name
- CITY, STATE, COUNTRY: Location information
- LATITUDE, LONGITUDE: Geographic coordinates

### Flights
Contains detailed flight information including:
- Date/time information (YEAR, MONTH, DAY, DAY_OF_WEEK)
- Flight identifiers (AIRLINE, FLIGHT_NUMBER, TAIL_NUMBER)
- Origin/destination (ORIGIN_AIRPORT, DESTINATION_AIRPORT)
- Departure/arrival times and delays
- Flight duration and distance metrics
- Cancellation and diversion indicators
- Delay category breakdowns

## Key Findings

The EDA script provides:
- Missing value analysis with recommended handling strategies
- Descriptive statistics for all numeric columns
- Temporal patterns (day of week analysis)
- Airline and airport statistics
- Delay patterns and distributions
- Flight cancellation rates

## Visualizations

![Flight EDA](./visualizations/flight_eda.png)