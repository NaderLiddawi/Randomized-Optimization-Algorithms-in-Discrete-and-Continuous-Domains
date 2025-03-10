
-------------------
Assignment 2 README
-------------------


------------------------------------------
Code and Data Reproducibility Instructions
------------------------------------------


## Overview
This repository contains the implementation for random optimization algorithms in discrete and continuous domains. The primary script, `main.py`, orchestrates data preprocessing, randomized optimization using three search algorithms—Randomized Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithm (GA)—and their application to both discrete optimization problems (FourPeaks and the Traveling Salesman Problem) as well as neural network weight tuning for a marketing campaign classification task. The code leverages Bayesian Optimization to tune hyperparameters and outputs evaluation metrics, plots, and CSV summaries to facilitate comprehensive analysis. These instructions are designed for reproducibility on a standard Linux machine, though they are equally applicable on Windows with minor adjustments (e.g., virtual environment activation commands).


## Directory Structure
- **README.md**: This file, which provides comprehensive instructions for running the code.
- **main.py**: The main Python script containing the complete implementation.
- **marketing_campaign.csv**: CSV file containing the marketing campaign dataset (expected to be tab-delimited and encoded in 'latin1').
- **results/**: Directory created at runtime to store output files such as plots and CSV summaries from hyperparameter searches.
- **requirements.txt** (optional): A file listing all required Python packages.


## Dependencies and Setup
The implementation requires **Python 3.8** or higher (preferably **Python 3.11**). The following libraries are used throughout the code:
- **numpy**: For numerical computations and array operations.
- **pandas**: For data manipulation and CSV file handling.
- **matplotlib**: For plotting graphs and visualizations.
- **scikit-learn**: For machine learning functionalities including model evaluation and preprocessing.
- **mlrose-hiive**: For implementing randomized optimization algorithms.
- **bayesian-optimization**: For tuning hyperparameters via Bayesian Optimization.
- **imbalanced-learn (imblearn)**: For handling imbalanced datasets (using SMOTEENN, SMOTE, and EditedNearestNeighbours).
- **tabulate**: For formatting output tables in the console.
- **joblib**: Used by mlrose-hiive for parallel processing and caching.


To install these dependencies, open a command prompt (on Windows) or terminal (on Linux) and run:

pip install numpy pandas matplotlib scikit-learn bayesian-optimization imbalanced-learn tabulate joblib==1.1.0 mlrose-hiive==2.2.4


Since a `requirements.txt` file is included in the repository, you may also install all required packages with:

   pip install -r requirements.txt

For isolation and to avoid dependency conflicts, it is recommended to create a Python virtual environment. On Windows, you can set one up using:

   python -m venv env
   env\Scripts\activate

On Linux, the activation command would be:

   python3 -m venv env
   source env/bin/activate


Data File:
------------
The execution of `main.py` depends on one CSV file that must be available in the same directory:

   - **marketing_campaign.csv:** Contains the marketing campaign dataset. This file is read using a tab delimiter (`\t`) and assumes a 'latin1' encoding.
   
   
Please ensure that this file is present in the repository’s root directory alongside `main.py`.


Running the Code:
---------------------
To execute the code and reproduce the results, follow these detailed steps:

   1. **Clone the Repository:**
      - Open a command prompt (on Windows) or terminal (on Linux).
      - Run:
            git clone https://github.com/NaderLiddawi/Randomized-Optimization-Algorithms-in-Discrete-and-Continuous-Domains.git

   2. **Set Up the Python Environment:**
      - Create a virtual environment:
            python -m venv env
      - Activate the virtual environment:
            env\Scripts\activate   (on Windows)
         or
            source env/bin/activate   (on Linux)

   3. **Install Dependencies:**
      - Install all required packages:
            pip install -r requirements.txt
         (or install packages individually as listed above).

   4. **Verify Data File:**
      - Confirm that `marketing_campaign.csv` is located in the same directory as `main.py`.

   5. **Run the Script:**
      - Execute the main Python script:
            python main.py
      - The script will preprocess the datasets, perform model training and hyperparameter tuning, and output results to both the console and the `results` directory. 




Data Availability and Reproducibility:
-------------------------------------------
The code and datasets have been prepared to ensure full reproducibility. All preprocessing steps (including label encoding, scaling, and handling class imbalance), training routines, hyperparameter searches, and evaluation metrics are documented both within the code and in this README file. All modifications are tracked in the commit history.


Research Report:
---------------------------------------------------
A research report that analyzes the code output rigorously is included in the same directory as the code. 
   

Additional Notes:
--------------------

- Please ensure the correct aforementioned versions of joblib and mlrose-hiive are installed as there are dependency conflicts otherwise.

- Because of the nature of Randomized Optimization algorithms in continuous space, the script might take over 30 minutes to run on standard CPUs.
 
- This repository has been designed with reproducibility in mind (RANDOM_SEED=42); every effort has been made to ensure that following these instructions will yield the same results as those reported in the assignment but for the runtime. Runtime variability is due to hardware quality, background processes and other factors. 


-------------
End of README
-------------
