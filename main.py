import os
import time
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from tabulate import tabulate

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# Bayesian Optimization library
from bayes_opt import BayesianOptimization

# imbalanced-learn library for refined handling of borderline/noise samples in imbalanced data
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL SETTINGS & CONSTANTS
# =============================================================================
GLOBAL_SEED = 42

# -----------------------------------------------------------------------------
# FROZEN_NN_CONFIG is used only for the Baseline (gradient_descent) method:
#  - 'activation' and 'hidden_layer_sizes' can be inherited by random opt NNs
#  - 'alpha' and 'learning_rate_init' are *not* for random optimization.
# -----------------------------------------------------------------------------
FROZEN_NN_CONFIG = {
    "params_activation": "sigmoid",
    "params_hidden_layer_sizes": [100],
    "params_alpha": 0.0002463768595899747,
    "params_learning_rate_init": 0.00582938454299474
}

MARKETING_CAMPAIGN_PATH = "marketing_campaign.csv"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
BASE_OUTPUT_DIR = "results"
CSV_DIR = os.path.join(BASE_OUTPUT_DIR, "csv_files")
IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# =============================================================================
# FILE SAVING HELPERS
# =============================================================================
def save_table(df, filename, header="", category=""):
    """
    Appends the DataFrame to a CSV file in a categorized folder.
    This function ensures reproducibility by creating necessary directories.
    """
    output_folder = os.path.join(CSV_DIR, category) if category else CSV_DIR
    os.makedirs(output_folder, exist_ok=True)
    full_path = os.path.join(output_folder, filename)
    with open(full_path, "a") as f:
        if header:
            f.write(header + "\n")
        df.to_csv(f, index=False)
        f.write("\n\n")


def save_plot(fig, filename, category=""):
    """
    Saves a matplotlib figure into a categorized folder.
    This helps organize convergence and performance plots under 'results/images'.
    """
    output_folder = os.path.join(IMG_DIR, category) if category else IMG_DIR
    os.makedirs(output_folder, exist_ok=True)
    full_path = os.path.join(output_folder, filename)
    fig.savefig(full_path, dpi=150)
    plt.close(fig)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def set_global_seed(seed_value):
    """
    Sets the seed for Python's random module and numpy to ensure reproducible
    experiments across multiple runs.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)


def preprocess_marketing_campaign(file_path):
    """
    Loads the marketing dataset, drops missing values,
    extracts 'Response' (binary classification) as y, and applies
    one-hot encoding to categorical features. This ensures that
    categorical data is properly transformed for neural network input.
    """
    full_data = pd.read_csv(file_path, delimiter='\t', encoding='latin1').dropna()
    y = full_data['Response'].astype(int)
    X = full_data.drop(columns=['ID', 'Dt_Customer', 'Response'], errors='ignore')

    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = pd.DataFrame(
            encoder.fit_transform(X[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols)
        )
        X = pd.concat([X[num_cols].reset_index(drop=True),
                       X_cat.reset_index(drop=True)], axis=1)
    return X, y


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_fitness_vs_iteration(all_curves, problem_name, algorithm_name, problem_size, maximize=True):
    """
    Plots the fitness curves across multiple runs (e.g., different seeds).
    We also plot the average ± std dev to visualize variation in performance.
    """
    plt.figure(figsize=(7, 5))
    processed_curves = []

    for i, curve in enumerate(all_curves):
        # If curve is 2D, first column is fitness. Otherwise, it's already 1D.
        if curve.ndim > 1:
            fitness_values = curve[:, 0]
        else:
            fitness_values = curve
        processed_curves.append(fitness_values)
        plt.plot(range(len(fitness_values)), fitness_values, alpha=0.4, label=f"Seed {i + 1}")

    # Compute and plot average ± std
    min_len = min(len(c) for c in processed_curves)
    truncated = [c[:min_len] for c in processed_curves]
    truncated = np.array(truncated)
    avg_curve = np.mean(truncated, axis=0)
    std_curve = np.std(truncated, axis=0)
    iters = np.arange(min_len)

    plt.plot(iters, avg_curve, 'k-', linewidth=2, label='Average')
    plt.fill_between(iters, avg_curve - std_curve, avg_curve + std_curve, color='gray', alpha=0.3, label='Std Dev')

    plt.title(f"{problem_name} (Size={problem_size}, {algorithm_name}) - Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness" if maximize else "Distance")
    plt.legend()
    plt.grid(True)

    png_name = f"{problem_name}_{problem_size}_{algorithm_name}_fitness_vs_iteration.png"
    category = f"{problem_name}/{algorithm_name}"
    save_plot(plt.gcf(), png_name, category=category)


def plot_fitness_vs_problem_size(results_dict, algo_name, maximize=True):
    """
    Plots the best achieved fitness vs. problem size. Helps visualize
    how an algorithm scales with growing complexity.
    """
    sizes = sorted(results_dict.keys())
    best_fitness = [results_dict[s] for s in sizes]

    plt.figure(figsize=(7, 5))
    plt.plot(sizes, best_fitness, marker="o", linestyle="-", label=algo_name)
    plt.xlabel("Problem Size")
    plt.ylabel("Best Fitness" if maximize else "Best Distance")
    plt.title(f"{algo_name}: Fitness vs Problem Size")
    plt.grid(True)
    plt.legend()

    png_name = f"Fitness_vs_ProblemSize_{algo_name}.png"
    category = f"{algo_name}/fitness_vs_problem_size"
    save_plot(plt.gcf(), png_name, category=category)


def plot_fitness_vs_time(evaluation_times, algorithm_name, problem_name, maximize=True):
    """
    Plots final fitness vs wall-clock time, and function evaluations vs time,
    giving insight into computational cost vs performance trade-offs.
    """
    param_dict = {}
    for entry in evaluation_times:
        # Skip entries with no valid curve.
        if entry.get("curve") is None:
            continue
        key = entry["params"]
        if key not in param_dict:
            param_dict[key] = {
                'elapsed': [],
                'fitness': [],
                'fevals': 0
            }
        param_dict[key]['elapsed'].append(entry["elapsed"])
        # curve is typically the final array of fitness values; we take the last element
        param_dict[key]['fitness'].append(entry["curve"][-1])
        param_dict[key]['fevals'] += entry.get("fevals", 1)

    times = [np.mean(param_dict[k]["elapsed"]) for k in param_dict]
    fitness_vals = [np.mean(param_dict[k]["fitness"]) for k in param_dict]
    fevals = [param_dict[k]["fevals"] for k in param_dict]

    # Plot final fitness vs time
    plt.figure(figsize=(7, 5))
    plt.scatter(times, fitness_vals, alpha=0.6, label=algorithm_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Final Fitness" if maximize else "Final Distance")
    plt.title(f"{problem_name} ({algorithm_name}) - Final Fitness vs Time")
    plt.legend()
    plt.grid(True)

    png_name = f"Fitness_vs_Time_{problem_name}_{algorithm_name}.png"
    category = f"{problem_name}/{algorithm_name}/fitness_vs_time"
    save_plot(plt.gcf(), png_name, category=category)

    # Plot function evaluations vs time
    plt.figure(figsize=(7, 5))
    plt.scatter(times, fevals, alpha=0.6, color='orange', label='F-Evals')
    plt.xlabel("Time (s)")
    plt.ylabel("Function Evaluations")
    plt.title(f"{problem_name} ({algorithm_name}) - F-Evals vs Time")
    plt.legend()
    plt.grid(True)

    png_name = f"Fevals_vs_Time_{problem_name}_{algorithm_name}.png"
    category = f"{problem_name}/{algorithm_name}/fevals_vs_time"
    save_plot(plt.gcf(), png_name, category=category)


def plot_runtime_vs_problem_size(problem_sizes, runtimes, algo_name):
    """
    Plots runtime vs problem size. Useful to observe how computational cost
    escalates with problem complexity for a given algorithm.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(problem_sizes, runtimes, 'o-', label=f"{algo_name} Runtime")
    plt.xlabel("Problem Size")
    plt.ylabel("Runtime (s)")
    plt.title(f"{algo_name}: Runtime vs Problem Size")
    plt.grid(True)
    plt.legend()

    png_name = f"Runtime_vs_ProblemSize_{algo_name}.png"
    category = f"{algo_name}/runtime_vs_problem_size"
    save_plot(plt.gcf(), png_name, category=category)


def plot_train_vs_test_performance(train_scores, test_scores, algo_name):
    """
    Plots train vs test performance (F1) after each fold or iteration,
    illustrating potential overfitting or underfitting patterns.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(train_scores, 'o-', label="Train F1")
    plt.plot(test_scores, 'o-', label="Test F1")
    plt.xlabel("Fold/Iteration")
    plt.ylabel("F1 Score")
    plt.title(f"{algo_name} - Train vs Test F1")
    plt.legend()
    plt.grid(True)

    png_name = f"Train_vs_Test_{algo_name}.png"
    category = f"NeuralNetworks/train_vs_test"
    save_plot(plt.gcf(), png_name, category=category)


def plot_fevals_vs_iteration(curve, problem_name, algorithm_name, problem_size, pop_size=None):
    """
    Plots the function evaluations vs iterations.
    Assumes each iteration corresponds to one function evaluation.
    """
    # Check if curve is None; if so, skip plotting.
    if curve is None:
        print(f"Warning: No fitness curve available for {problem_name} {algorithm_name} (size={problem_size}). Skipping plot_fevals_vs_iteration.")
        return

    plt.figure(figsize=(7, 5))
    iterations = np.arange(len(curve))
    if algorithm_name == "GA" and pop_size:
        fevals = iterations * pop_size
    else:
        fevals = iterations  # assuming one feval per iteration for RHC and SA

    plt.plot(iterations, fevals, 'r-', linewidth=2, label='F-Evals')
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.title(f"{problem_name} ({algorithm_name}) - F-Evals vs Iteration")
    plt.legend()
    plt.grid(True)
    png_name = f"Fevals_vs_Iteration_{problem_name}_{problem_size}_{algorithm_name}.png"
    category = f"{problem_name}/{algorithm_name}/fevals_vs_iteration"
    save_plot(plt.gcf(), png_name, category=category)


def plot_multiple_fitness_curves(curves_dict, problem_name, algorithm_name, maximize=True):
    """
    Plots multiple fitness curves on the same graph.
    Parameters:
      curves_dict: a dictionary where keys are problem sizes and values are the representative fitness curves.
      problem_name: string indicating the problem (e.g., "TSP" or "FourPeaks").
      algorithm_name: string indicating the algorithm (e.g., "RHC").
      maximize: boolean flag to determine y-axis label.
    """
    plt.figure(figsize=(7, 5))
    for size, curve in curves_dict.items():
        # Skip if curve is None.
        if curve is None:
            print(f"Warning: No fitness curve available for {problem_name} {algorithm_name} (size={size}). Skipping.")
            continue
        if curve.ndim > 1:
            fitness_values = curve[:, 0]
        else:
            fitness_values = curve
        plt.plot(range(len(fitness_values)), fitness_values, label=f"Size={size}")

    plt.title(f"{problem_name} - {algorithm_name}: Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness" if maximize else "Distance")
    plt.legend()
    plt.grid(True)
    png_name = f"{problem_name}_{algorithm_name}_combined_fitness_vs_iteration.png"
    save_plot(plt.gcf(), png_name, category=f"{problem_name}/{algorithm_name}")



def generate_summary_table(algorithm_results, filename):
    """
    Prints a consolidated summary table in the console (via tabulate)
    and then saves it to a single CSV for easy reference.
    """
    summary_df = pd.DataFrame(algorithm_results)
    print("\n=== Summary of Algorithm Performance ===")
    print(tabulate(summary_df, headers="keys", tablefmt="github"))
    summary_df.to_csv(filename, index=False)


# =============================================================================
# SA / DECAY HELPERS
# =============================================================================
from mlrose_hiive import ExpDecay, GeomDecay, ArithDecay


def interpret_sa_schedule(index):
    """
    Interprets the schedule index for SA:
      0 -> "exp"
      1 -> "geom"
      2 -> "arith"
    """
    if index == 0:
        return "exp"
    elif index == 1:
        return "geom"
    return "arith"


def schedule_object(schedule_index, init_temp, decay):
    """
    Based on the schedule_index, returns an ExpDecay, GeomDecay, or ArithDecay object.
    """
    schedule_name = interpret_sa_schedule(schedule_index)
    if schedule_name == "exp":
        return ExpDecay(init_temp=init_temp, exp_const=decay, min_temp=0.001)
    elif schedule_name == "geom":
        return GeomDecay(init_temp=init_temp, decay=decay, min_temp=0.001)
    else:
        return ArithDecay(init_temp=init_temp, decay=decay, min_temp=0.001)


###############################################################################
#  1) RHC, SA, GA for Discrete Problems (Section 3.1)
###############################################################################
def search_discrete_rhc(problem, n_iter_bayes=10):
    """
    Bayesian optimization for Random Hill Climb (RHC).
    - Encouragement: Try more init_points or n_iter if results are suboptimal.
    - Also consider changing the ranges of 'pbounds' if needed for your domain.
    """
    from bayes_opt import BayesianOptimization
    evaluation_times = []
    best_curves = []

    def objective(num_restarts, max_iterations, max_attempts):
        # Remove acronyms, keep code descriptive
        restarts = int(round(num_restarts))
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))

        start_time = time.time()
        _, fitness, curve = mlrose.random_hill_climb(
            problem,
            restarts=restarts,
            max_iters=iterations,
            max_attempts=attempts,
            curve=True,
            random_state=GLOBAL_SEED
        )
        elapsed = time.time() - start_time
        # [FIXED: fevals logging]
        evaluation_times.append({
            'params': (restarts, iterations, attempts),
            'elapsed': elapsed,
            'curve': curve,
            'fevals': len(curve) if curve is not None else 1  # changed from 1 to len(curve)
        })
        return fitness

    pbounds = {
        'num_restarts': (0, 20),  # Reduce max restarts to 20; more restarts can slow training unnecessarily
        'max_iterations': (100, 500),  # Reduce the upper bound; beyond 500 has diminishing returns
        'max_attempts': (5, 50)  # Keep as is; large max_attempts can cause early convergence issues
    }
    print("\n[Bayes Opt] Searching RHC hyperparameters...")
    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)

    # For improved performance, you can increase init_points or n_iter
    # e.g., init_points=5, n_iter=15
    bo.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    results_rows = []
    for res in sorted(bo.res, key=lambda x: x['target'], reverse=True)[:5]:
        params = res['params']
        restarts = int(round(params['num_restarts']))
        iterations = int(round(params['max_iterations']))
        attempts = int(round(params['max_attempts']))
        matching_runs = [e for e in evaluation_times if e['params'] == (restarts, iterations, attempts)]
        times = [m['elapsed'] for m in matching_runs]
        curves = [m['curve'] for m in matching_runs]
        avg_elapsed = np.mean(times) if times else 0.0

        results_rows.append({
            'RHC_NumRestarts': restarts,
            'RHC_MaxIterations': iterations,
            'RHC_MaxAttempts': attempts,
            'Fitness': round(res['target'], 4),
            'Elapsed_Time': round(avg_elapsed, 4),
            'Func_Evals': sum(m.get("fevals", 1) for m in matching_runs)
        })
        if curves:
            best_curves.append(curves[0])

    df = pd.DataFrame(results_rows)
    return df, best_curves, evaluation_times


def search_discrete_sa(problem, n_iter_bayes=10):
    """
    Bayesian optimization for Simulated Annealing (SA).
    - We can try different schedules: ExpDecay, GeomDecay, ArithDecay
    - Also can tweak ranges for temperature_decay if results are poor.
    """
    from bayes_opt import BayesianOptimization
    evaluation_times = []
    best_curves = []

    def objective(schedule_index, initial_temperature, max_iterations, max_attempts, temperature_decay):
        # Remove acronyms for clarity
        index = int(round(schedule_index))
        index = max(min(index, 2), 0)
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))
        schedule = schedule_object(index, initial_temperature, temperature_decay)

        start_time = time.time()
        _, fitness, curve = mlrose.simulated_annealing(
            problem,
            schedule=schedule,
            max_iters=iterations,
            max_attempts=attempts,
            curve=True,
            random_state=GLOBAL_SEED
        )
        elapsed = time.time() - start_time
        # [FIXED: fevals logging]
        evaluation_times.append({
            'params': (index, initial_temperature, iterations, attempts, temperature_decay),
            'elapsed': elapsed,
            'curve': curve,
            'fevals': len(curve) if curve is not None else 1  # changed from 1 to len(curve)
        })
        return fitness

    pbounds = {
        'schedule_index': (0, 2),  # Keep as is; allows testing of exponential, geometric, and arithmetic decay
        'initial_temperature': (1, 20),  # Reduce upper bound; very high values (>20) lead to excessive exploration
        'max_iterations': (100, 500),  # Reduce to 500 to prevent wasted iterations
        'max_attempts': (5, 50),  # Reduce upper bound; 100 was excessive
        'temperature_decay': (0.01, 0.5)  # Reduce decay rate max to 0.5 for smoother cooling
    }

    print("\n[Bayes Opt] Searching SA hyperparameters...")
    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)
    bo.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    rows = []
    for res in sorted(bo.res, key=lambda x: x['target'], reverse=True)[:5]:
        p = res['params']
        idx = int(round(p['schedule_index']))
        idx = max(min(idx, 2), 0)
        init_temp = p['initial_temperature']
        iterations = int(round(p['max_iterations']))
        attempts = int(round(p['max_attempts']))
        decay_rate = p['temperature_decay']
        schedule_name = interpret_sa_schedule(idx)

        run_matches = [
            e for e in evaluation_times if e['params'] == (idx, init_temp, iterations, attempts, decay_rate)
        ]
        times = [m['elapsed'] for m in run_matches]
        curves = [m['curve'] for m in run_matches]
        avg_elapsed = np.mean(times) if times else 0.0

        rows.append({
            'SA_ScheduleType': schedule_name,
            'SA_InitialTemperature': round(init_temp, 4),
            'SA_MaxIterations': iterations,
            'SA_MaxAttempts': attempts,
            'SA_TemperatureDecay': round(decay_rate, 4),
            'Fitness': round(res['target'], 4),
            'Elapsed_Time': round(avg_elapsed, 4),
            'Func_Evals': sum(m.get("fevals", 1) for m in run_matches)
        })
        if curves:
            best_curves.append(curves[0])
    df = pd.DataFrame(rows)
    return df, best_curves, evaluation_times


def search_discrete_ga(problem, n_iter_bayes=10):
    """
    Bayesian optimization for Genetic Algorithm (GA).
    - population_size, mutation_probability, crossover_probability, etc.
    """
    from bayes_opt import BayesianOptimization
    evaluation_times = []
    best_curves = []

    def objective(population_size, mutation_prob, crossover_prob, max_iterations, max_attempts):
        # Replace acronyms with clarity
        pop_size = int(round(population_size))
        mutation_probability = mutation_prob
        crossover_probability = crossover_prob
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))

        start_time = time.time()
        _, fitness, curve = mlrose.genetic_alg(
            problem,
            pop_size=pop_size,
            mutation_prob=mutation_probability,
            pop_breed_percent=crossover_probability,
            max_iters=iterations,
            max_attempts=attempts,
            curve=True,
            random_state=GLOBAL_SEED
        )
        elapsed = time.time() - start_time
        # [FIXED: fevals logging]
        evaluation_times.append({
            'params': (pop_size, mutation_probability, crossover_probability, iterations, attempts),
            'elapsed': elapsed,
            'curve': curve,
            'fevals': len(curve) if curve is not None else 1  # changed from 1 to len(curve)
        })
        return fitness

    pbounds = {
        'population_size': (30, 200),  # Reduce max population size; 300 is excessive
        'mutation_prob': (0.01, 0.3),  # Increase upper bound slightly; GA benefits from more mutation diversity
        'crossover_prob': (0.4, 0.9),  # Increase lower bound; low crossover values can slow convergence
        'max_iterations': (50, 200),  # Reduce upper bound; excessive iterations provide diminishing returns
        'max_attempts': (5, 30)  # Reduce upper bound to avoid overfitting to poor solutions
    }
    print("\n[Bayes Opt] Searching GA hyperparameters...")
    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)
    bo.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    rows = []
    for res in sorted(bo.res, key=lambda x: x['target'], reverse=True)[:5]:
        p = res['params']
        pop_size = int(round(p['population_size']))
        mutation_probability = p['mutation_prob']
        crossover_probability = p['crossover_prob']
        iterations = int(round(p['max_iterations']))
        attempts = int(round(p['max_attempts']))

        run_matches = [
            e for e in evaluation_times if e['params'] == (pop_size, mutation_probability,
                                                           crossover_probability, iterations, attempts)
        ]
        times = [m['elapsed'] for m in run_matches]
        curves = [m['curve'] for m in run_matches]
        avg_elapsed = np.mean(times) if times else 0.0

        rows.append({
            'GA_PopulationSize': pop_size,
            'GA_MutationProb': round(mutation_probability, 4),
            'GA_CrossoverProb': round(crossover_probability, 4),
            'GA_MaxIterations': iterations,
            'GA_MaxAttempts': attempts,
            'Fitness': round(res['target'], 4),
            'Elapsed_Time': round(avg_elapsed, 4),
            'Func_Evals': sum(m.get("fevals", 1) for m in run_matches)
        })
        if curves:
            best_curves.append(curves[0])
    df = pd.DataFrame(rows)
    return df, best_curves, evaluation_times


#########################################################################
#  2) NEURAL NETWORK TRAINING HELPER FUNCTIONS & Search (Section 3.2)
#########################################################################
def run_model_and_get_test_scores(params_dict, X_train, y_train, X_test, y_test):
    """
    Train a neural network using the provided hyperparameters. Return:
      - test_f1_micro
      - test_auc (ROC)
      - test_f1_macro
      - test_balanced_acc

    NOTE: We only carry forward 'activation' and 'hidden_nodes' from the FROZEN_NN_CONFIG
    for random optimization methods. 'alpha' and 'learning_rate' are used solely for
    baseline backprop (gradient_descent).
    """
    algo = params_dict["algorithm"]
    # Common parameters for any NN approach
    final_params = {
        "activation": FROZEN_NN_CONFIG["params_activation"],
        "hidden_nodes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        "early_stopping": True,
        "clip_max": 5.0,
        "bias": True,
        "is_classifier": True,
        "curve": True,  # We'll keep this as is; if you'd like to track curves, set to True for random opt
        "random_state": GLOBAL_SEED,
        "algorithm": algo
    }

    # For the baseline (GD), we include alpha & learning_rate from FROZEN_NN_CONFIG
    if algo == "gradient_descent":
        final_params["learning_rate"] = FROZEN_NN_CONFIG["params_learning_rate_init"]
        final_params["max_iters"] = 500

    elif algo == "random_hill_climb":
        final_params["restarts"] = params_dict.get("RHC_NumRestarts", 0)
        final_params["max_iters"] = params_dict.get("RHC_MaxIterations", 100)
        final_params["max_attempts"] = params_dict.get("RHC_MaxAttempts", 50)

    elif algo == "simulated_annealing":
        final_params["max_iters"] = params_dict.get("SA_MaxIterations", 100)
        final_params["max_attempts"] = params_dict.get("SA_MaxAttempts", 50)
        schedule_type = params_dict.get("SA_ScheduleType", "geom")
        initial_temp = params_dict.get("SA_InitialTemperature", 10)
        decay_rate = params_dict.get("SA_TemperatureDecay", 0.99)
        if schedule_type == "exp":
            final_params["schedule"] = ExpDecay(init_temp=initial_temp, exp_const=decay_rate, min_temp=0.001)
        elif schedule_type == "geom":
            final_params["schedule"] = GeomDecay(init_temp=initial_temp, decay=decay_rate, min_temp=0.001)
        else:
            final_params["schedule"] = ArithDecay(init_temp=initial_temp, decay=decay_rate, min_temp=0.001)

    elif algo == "genetic_alg":
        final_params["pop_size"] = params_dict.get("GA_PopulationSize", 50)
        final_params["mutation_prob"] = params_dict.get("GA_MutationProb", 0.1)
        final_params["max_iters"] = params_dict.get("GA_MaxIterations", 20)
        final_params["max_attempts"] = params_dict.get("GA_MaxAttempts", 10)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    model = mlrose.NeuralNetwork(**final_params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_probs = model.predicted_probs

    if y_test_probs.ndim == 2 and y_test_probs.shape[1] == 2:
        pos_probs = y_test_probs[:, 1]
    else:
        pos_probs = y_test_probs

    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    auc_score = roc_auc_score(y_test, pos_probs)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    return test_f1_micro, auc_score, test_f1_macro, bal_acc


def run_nn_frozen_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Baseline NN approach (Gradient Descent). We do:
      - 5-fold CV on (X_train, y_train)
      - Train once more on entire (X_train, y_train)
      - Evaluate on X_test
      Also, track convergence behavior via the fitness curves.
    """
    final_params = {
        "activation": FROZEN_NN_CONFIG["params_activation"],
        "hidden_nodes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        # alpha & learning_rate only apply here
        "learning_rate": FROZEN_NN_CONFIG["params_learning_rate_init"],
        "algorithm": "gradient_descent",
        "early_stopping": True,
        "clip_max": 5.0,
        "bias": True,
        "is_classifier": True,
        "curve": True,  # Ensure we track the fitness curve
        "random_state": GLOBAL_SEED
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    cv_scores = []
    cv_times = []
    cv_fevals = []  # NEW: to record the number of function evaluations per fold

    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)

    # 1) Cross-validation
    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        X_fold_train, y_fold_train = X_train_np[train_idx], y_train_np[train_idx]
        X_fold_val, y_fold_val = X_train_np[val_idx], y_train_np[val_idx]

        start_time = time.time()
        model = mlrose.NeuralNetwork(**final_params)
        model.fit(X_fold_train, y_fold_train)
        preds_val = model.predict(X_fold_val)
        fold_f1 = f1_score(y_fold_val, preds_val, average='micro')
        cv_scores.append(fold_f1)
        cv_times.append(time.time() - start_time)
        # Record the number of evaluations (length of fitness_curve)
        if hasattr(model, 'fitness_curve') and model.fitness_curve is not None:
            cv_fevals.append(len(model.fitness_curve))
        else:
            cv_fevals.append(0)

    cv_f1 = np.mean(cv_scores)
    cv_elapsed = np.mean(cv_times)
    avg_cv_fevals = np.mean(cv_fevals)  # NEW: average number of evaluations in CV

    # 2) Retrain on entire train set and track convergence
    start_time = time.time()
    final_model = mlrose.NeuralNetwork(**final_params)
    final_model.fit(X_train_np, y_train_np)

    if hasattr(final_model, 'fitness_curve') and final_model.fitness_curve is not None:
        iterations = np.arange(len(final_model.fitness_curve))
        fevals = iterations  # assuming one function evaluation per iteration
        plt.figure(figsize=(7, 5))
        plt.plot(iterations, fevals, 'r-', linewidth=2, label='F-Evals')
        plt.xlabel("Iteration")
        plt.ylabel("Function Evaluations")
        plt.title("NN (Baseline) - F-Evals vs Iteration")
        plt.legend()
        plt.grid(True)
        save_plot(plt.gcf(), "Fevals_vs_Iteration_NN_Baseline.png", category="NeuralNetworks")

    train_time = time.time() - start_time
    # Record final model's function evaluations
    if hasattr(final_model, 'fitness_curve') and final_model.fitness_curve is not None:
        test_fevals = len(final_model.fitness_curve)
    else:
        test_fevals = 0

    # 3) Evaluate on test set
    test_preds = final_model.predict(X_test)
    test_probs = final_model.predicted_probs
    if test_probs.ndim == 2 and test_probs.shape[1] == 2:
        pos_probs = test_probs[:, 1]
    else:
        pos_probs = test_probs

    test_f1_micro = f1_score(y_test, test_preds, average='micro')
    test_auc = roc_auc_score(y_test, pos_probs)
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)

    total_elapsed = cv_elapsed + train_time
    summary = {
        "NN_params_activation": FROZEN_NN_CONFIG["params_activation"],
        "NN_params_hidden_layer_sizes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        "NN_params_alpha": FROZEN_NN_CONFIG["params_alpha"],
        "NN_params_learning_rate_init": FROZEN_NN_CONFIG["params_learning_rate_init"],
        "CV_F1": round(cv_f1, 4),
        "CV_Fevals": round(avg_cv_fevals, 2),  # average CV function evaluations
        "Test_F1 (Holdout)": round(test_f1_micro, 4),
        "Test_ROC_AUC (Holdout)": round(test_auc, 4),
        "Test_F1_Macro": round(test_f1_macro, 4),
        "Test_Balanced_Accuracy": round(test_bal_acc, 4),
        "Test_Fevals": test_fevals,  # function evaluations for final training
        "Elapsed_Time": round(total_elapsed, 4)
    }

    return summary, cv_scores


def cross_validate_nn(X_train, y_train, param_dict):
    """
    A 5-fold cross-validation utility specifically for RHC, SA, or GA.
    The 'param_dict' includes the 'algorithm' key plus relevant hyperparams.
    We compute F1 on both training and validation folds to plot train vs test performance.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    train_scores = []
    test_scores = []

    X_np = np.asarray(X_train)
    y_np = np.asarray(y_train)

    for train_idx, val_idx in skf.split(X_np, y_np):
        X_fold_train, y_fold_train = X_np[train_idx], y_np[train_idx]
        X_fold_val, y_fold_val = X_np[val_idx], y_np[val_idx]

        # Evaluate on train fold
        f1_train, _, _, _ = run_model_and_get_test_scores(param_dict, X_fold_train, y_fold_train,
                                                          X_fold_train, y_fold_train)
        train_scores.append(f1_train)

        # Evaluate on validation fold
        f1_val, _, _, _ = run_model_and_get_test_scores(param_dict, X_fold_train, y_fold_train,
                                                        X_fold_val, y_fold_val)
        test_scores.append(f1_val)

    return train_scores, test_scores


###############################################################################
#  3) NN Hyperparam Tuning (RHC, SA, GA)
#     (Detailed code with bullet points and commentary)
###############################################################################
def search_nn_rhc(X_train, y_train, X_test, y_test, n_iter_bayes=5):
    """
    Bayesian optimization for RHC hyperparameters in the Neural Network setting.
    - We only pass activation & hidden_nodes from FROZEN_NN_CONFIG,
      ignoring alpha/learning_rate specifically for random optimization methods.
    """
    evaluation_times = []

    def cv_score(r_starts, max_iterations, max_attempts):
        restarts = int(round(r_starts))
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))

        start_time = time.time()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        scores = []
        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        for tr_idx, val_idx in skf.split(X_np, y_np):
            X_fold_train, y_fold_train = X_np[tr_idx], y_np[tr_idx]
            X_fold_val, y_fold_val = X_np[val_idx], y_np[val_idx]

            param_dict = {
                "algorithm": "random_hill_climb",
                "RHC_NumRestarts": restarts,
                "RHC_MaxIterations": iterations,
                "RHC_MaxAttempts": attempts
            }
            fold_f1, _, _, _ = run_model_and_get_test_scores(param_dict,
                                                             X_fold_train, y_fold_train,
                                                             X_fold_val, y_fold_val)
            scores.append(fold_f1)

        elapsed = time.time() - start_time
        fevals_count = len(scores)  # We did 5 folds

        # We no longer store a single-item curve; just keep curve=None here
        evaluation_times.append({
            'params': (restarts, iterations, attempts),
            'elapsed': elapsed,
            'fevals': fevals_count,
            'curve': None
        })
        return np.mean(scores)

    print("\n[Bayes Opt NN] Searching RHC hyperparameters for NN...")
    pbounds = {
        'r_starts': (0, 20),
        'max_iterations': (100, 500),
        'max_attempts': (5, 50)
    }
    optimizer = BayesianOptimization(f=cv_score, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)
    optimizer.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    rows = []
    # After Bayesian Optimization finishes, do a final run on full train set
    # to get the *actual* iteration-level fitness_curve.
    for res in sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:5]:
        p = res['params']
        restarts = int(round(p['r_starts']))
        iterations = int(round(p['max_iterations']))
        attempts = int(round(p['max_attempts']))

        # Evaluate holdout performance:
        f1_micro, auc_score, f1_macro, bal_acc = run_model_and_get_test_scores(
            {
                "algorithm": "random_hill_climb",
                "RHC_NumRestarts": restarts,
                "RHC_MaxIterations": iterations,
                "RHC_MaxAttempts": attempts
            },
            X_train, y_train, X_test, y_test
        )

        # Now do one final training run to record iteration-based curve
        start_time_final = time.time()
        final_params = {
            "activation": FROZEN_NN_CONFIG["params_activation"],
            "hidden_nodes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
            "algorithm": "random_hill_climb",
            "restarts": restarts,
            "max_iters": iterations,
            "max_attempts": attempts,
            "early_stopping": True,
            "clip_max": 5.0,
            "bias": True,
            "is_classifier": True,
            "curve": True,
            "random_state": GLOBAL_SEED
        }
        final_model = mlrose.NeuralNetwork(**final_params)
        final_model.fit(X_train, y_train)
        elapsed_final = time.time() - start_time_final

        if final_model.fitness_curve is not None:
            final_curve = final_model.fitness_curve
            final_fevals = len(final_curve)
        else:
            final_curve = np.array([0])
            final_fevals = 1

        # Store iteration-based data in evaluation_times
        evaluation_times.append({
            'params': (restarts, iterations, attempts),
            'elapsed': elapsed_final,
            'curve': final_curve,
            'fevals': final_fevals
        })

        rows.append({
            "Algorithm": "RHC",
            "RHC_NumRestarts": restarts,
            "RHC_MaxIterations": iterations,
            "RHC_MaxAttempts": attempts,
            "CV_F1": round(res['target'], 4),
            "Test_F1 (Holdout)": round(f1_micro, 4),
            "Test_ROC_AUC (Holdout)": round(auc_score, 4),
            "Test_F1_Macro": round(f1_macro, 4),
            "Test_Balanced_Accuracy": round(bal_acc, 4),
            "Elapsed_Time": round(elapsed_final, 4),
            "Func_Evals": final_fevals
        })

    return pd.DataFrame(rows), evaluation_times


def search_nn_sa(X_train, y_train, X_test, y_test, n_iter_bayes=5):
    """
    Bayesian optimization for SA hyperparameters in the NN context.
    - As recommended, we can try different schedules, initial temperatures, decay rates.
    """
    evaluation_times = []

    def cv_score(schedule_index, initial_temperature, max_iterations, max_attempts, temperature_decay):
        si = int(round(schedule_index))
        si = max(0, min(si, 2))
        init_temp = initial_temperature
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))
        decay = temperature_decay

        start_time = time.time()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        scores = []
        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        for tr_idx, val_idx in skf.split(X_np, y_np):
            X_fold_train, y_fold_train = X_np[tr_idx], y_np[tr_idx]
            X_fold_val, y_fold_val = X_np[val_idx], y_np[val_idx]

            schedule_name = interpret_sa_schedule(si)
            param_dict = {
                "algorithm": "simulated_annealing",
                "SA_ScheduleType": schedule_name,
                "SA_InitialTemperature": init_temp,
                "SA_MaxIterations": iterations,
                "SA_MaxAttempts": attempts,
                "SA_TemperatureDecay": decay
            }
            fold_f1, _, _, _ = run_model_and_get_test_scores(param_dict,
                                                             X_fold_train, y_fold_train,
                                                             X_fold_val, y_fold_val)
            scores.append(fold_f1)

        elapsed = time.time() - start_time
        fevals_count = len(scores)

        # We no longer store a single-item curve; just keep curve=None here
        evaluation_times.append({
            'params': (si, init_temp, iterations, attempts, decay),
            'elapsed': elapsed,
            'fevals': fevals_count,
            'curve': None
        })
        return np.mean(scores)

    print("\n[Bayes Opt NN] Searching SA hyperparameters for NN...")
    pbounds = {
        'schedule_index': (0, 2),
        'initial_temperature': (1, 20),
        'max_iterations': (100, 500),
        'max_attempts': (5, 50),
        'temperature_decay': (0.01, 0.5)
    }

    optimizer = BayesianOptimization(f=cv_score, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)
    optimizer.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    rows = []
    # After Bayesian Optimization, do a final run to get iteration-based curve
    for res in sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:5]:
        p = res['params']
        si = int(round(p['schedule_index']))
        si = max(0, min(si, 2))
        schedule_name = interpret_sa_schedule(si)
        init_temp = p['initial_temperature']
        iterations = int(round(p['max_iterations']))
        attempts = int(round(p['max_attempts']))
        decay = p['temperature_decay']

        param_dict = {
            "algorithm": "simulated_annealing",
            "SA_ScheduleType": schedule_name,
            "SA_InitialTemperature": init_temp,
            "SA_MaxIterations": iterations,
            "SA_MaxAttempts": attempts,
            "SA_TemperatureDecay": decay
        }
        f1_micro, auc_score, f1_macro, bal_acc = run_model_and_get_test_scores(
            param_dict, X_train, y_train, X_test, y_test
        )

        # Final training run for iteration-level curve
        start_time_final = time.time()
        final_params = {
            "activation": FROZEN_NN_CONFIG["params_activation"],
            "hidden_nodes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
            "algorithm": "simulated_annealing",
            "max_iters": iterations,
            "max_attempts": attempts,
            "early_stopping": True,
            "clip_max": 5.0,
            "bias": True,
            "is_classifier": True,
            "curve": True,
            "random_state": GLOBAL_SEED
        }
        # Set the schedule
        final_params["schedule"] = schedule_object(si, init_temp, decay)

        final_model = mlrose.NeuralNetwork(**final_params)
        final_model.fit(X_train, y_train)
        elapsed_final = time.time() - start_time_final

        if final_model.fitness_curve is not None:
            final_curve = final_model.fitness_curve
            final_fevals = len(final_curve)
        else:
            final_curve = np.array([0])
            final_fevals = 1

        evaluation_times.append({
            'params': (si, init_temp, iterations, attempts, decay),
            'elapsed': elapsed_final,
            'curve': final_curve,
            'fevals': final_fevals
        })

        rows.append({
            "NN_params_activation": FROZEN_NN_CONFIG["params_activation"],
            "NN_params_hidden_layer_sizes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
            "SA_ScheduleType": schedule_name,
            "SA_InitialTemperature": round(init_temp, 4),
            "SA_MaxIterations": iterations,
            "SA_MaxAttempts": attempts,
            "SA_TemperatureDecay": round(decay, 4),
            "CV_F1": round(res['target'], 4),
            "Test_F1 (Holdout)": round(f1_micro, 4),
            "Test_ROC_AUC (Holdout)": round(auc_score, 4),
            "Test_F1_Macro": round(f1_macro, 4),
            "Test_Balanced_Accuracy": round(bal_acc, 4),
            "Elapsed_Time": round(elapsed_final, 4),
            "Func_Evals": final_fevals
        })

    return pd.DataFrame(rows), evaluation_times


def search_nn_ga(X_train, y_train, X_test, y_test, n_iter_bayes=5):
    """
    Bayesian optimization for GA hyperparameters in the NN context.
    - population_size, mutation_probability, crossover_probability, etc.
    """
    evaluation_times = []

    def cv_score(population_size, mutation_prob, crossover_prob, max_iterations, max_attempts):
        pop_size = int(round(population_size))
        mutation_probability = mutation_prob
        crossover_probability = crossover_prob
        iterations = int(round(max_iterations))
        attempts = int(round(max_attempts))

        start_time = time.time()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
        scores = []
        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        for tr_idx, val_idx in skf.split(X_np, y_np):
            X_fold_train, y_fold_train = X_np[tr_idx], y_np[tr_idx]
            X_fold_val, y_fold_val = X_np[val_idx], y_np[val_idx]

            param_dict = {
                "algorithm": "genetic_alg",
                "GA_PopulationSize": pop_size,
                "GA_MutationProb": mutation_probability,
                "GA_CrossoverProb": crossover_probability,
                "GA_MaxIterations": iterations,
                "GA_MaxAttempts": attempts
            }
            fold_f1, _, _, _ = run_model_and_get_test_scores(param_dict,
                                                             X_fold_train, y_fold_train,
                                                             X_fold_val, y_fold_val)
            scores.append(fold_f1)

        elapsed = time.time() - start_time
        fevals_count = len(scores)

        # We no longer store a single-item curve; just keep curve=None here
        evaluation_times.append({
            'params': (pop_size, mutation_probability, crossover_probability, iterations, attempts),
            'elapsed': elapsed,
            'fevals': fevals_count,
            'curve': None
        })
        return np.mean(scores)

    # Define Bayesian optimization search space
    pbounds = {
        'population_size': (30, 200),
        'mutation_prob': (0.01, 0.3),
        'crossover_prob': (0.4, 0.9),
        'max_iterations': (50, 200),
        'max_attempts': (5, 30)
    }

    print("\n[Bayes Opt NN] Searching GA hyperparameters for NN...")
    optimizer = BayesianOptimization(f=cv_score, pbounds=pbounds, random_state=GLOBAL_SEED, verbose=1)
    optimizer.maximize(init_points=5, n_iter=n_iter_bayes - 5)

    rows = []
    # After Bayesian Optimization, do a final run to get iteration-based curve
    for res in sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:5]:
        p = res['params']
        pop_size = int(round(p['population_size']))
        mutation_probability = p['mutation_prob']
        crossover_probability = p['crossover_prob']
        iterations = int(round(p['max_iterations']))
        attempts = int(round(p['max_attempts']))

        # Evaluate on holdout test
        param_dict = {
            "algorithm": "genetic_alg",
            "GA_PopulationSize": pop_size,
            "GA_MutationProb": mutation_probability,
            "GA_CrossoverProb": crossover_probability,
            "GA_MaxIterations": iterations,
            "GA_MaxAttempts": attempts
        }
        f1_micro, auc_score, f1_macro, bal_acc = run_model_and_get_test_scores(
            param_dict, X_train, y_train, X_test, y_test
        )

        # Final training run for iteration-level curve
        start_time_final = time.time()
        final_params = {
            "activation": FROZEN_NN_CONFIG["params_activation"],
            "hidden_nodes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
            "algorithm": "genetic_alg",
            "pop_size": pop_size,
            "mutation_prob": mutation_probability,
            "max_iters": iterations,
            "max_attempts": attempts,
            "early_stopping": True,
            "clip_max": 5.0,
            "bias": True,
            "is_classifier": True,
            "curve": True,
            "random_state": GLOBAL_SEED
        }


        final_model = mlrose.NeuralNetwork(**final_params)
        final_model.fit(X_train, y_train)
        elapsed_final = time.time() - start_time_final

        if final_model.fitness_curve is not None:
            final_curve = final_model.fitness_curve
            final_fevals = len(final_curve)
        else:
            final_curve = np.array([0])
            final_fevals = 1

        evaluation_times.append({
            'params': (pop_size, mutation_probability, crossover_probability, iterations, attempts),
            'elapsed': elapsed_final,
            'curve': final_curve,
            'fevals': final_fevals
        })

        rows.append({
            "NN_params_activation": FROZEN_NN_CONFIG["params_activation"],
            "NN_params_hidden_layer_sizes": FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
            "GA_PopulationSize": pop_size,
            "GA_MutationProb": round(mutation_probability, 4),
            "GA_CrossoverProb": round(crossover_probability, 4),
            "GA_MaxIterations": iterations,
            "GA_MaxAttempts": attempts,
            "CV_F1": round(res['target'], 4),
            "Test_F1 (Holdout)": round(f1_micro, 4),
            "Test_ROC_AUC (Holdout)": round(auc_score, 4),
            "Test_F1_Macro": round(f1_macro, 4),
            "Test_Balanced_Accuracy": round(bal_acc, 4),
            "Elapsed_Time": round(elapsed_final, 4),
            "Func_Evals": final_fevals
        })

    return pd.DataFrame(rows), evaluation_times


###############################################################################
#  4) Discrete Optimization Orchestration
#  -> RHC, SA, GA on FourPeaks + TSP
###############################################################################
def run_ro_optimization_problems():
    """
    1) FourPeaks with sizes [30, 60, 90]
    2) TSP with sizes [10, 20, 30]
    - We use negative distance to turn TSP into a maximization problem.
    - We gather results, produce CSV output in subfolders,
      and produce relevant plots.
    """
    # Data structures to store best results for each size, for each algorithm
    four_peaks_fitness = {"RHC": {}, "SA": {}, "GA": {}}
    four_peaks_runtimes = {"RHC": {}, "SA": {}, "GA": {}}
    tsp_fitness = {"RHC": {}, "SA": {}, "GA": {}}
    tsp_runtimes = {"RHC": {}, "SA": {}, "GA": {}}

    # =======================
    # FourPeaks problem
    # =======================
    from mlrose_hiive import DiscreteOpt, FourPeaks
    fp_csv = "FourPeaks_Results.csv"
    fp_folder = os.path.join(CSV_DIR, "FourPeaks")
    os.makedirs(fp_folder, exist_ok=True)

    # Remove the existing CSV file if it exists to avoid permission errors.
    csv_path = os.path.join(fp_folder, fp_csv)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Create dictionary to store representative curves for combined plotting
    four_peaks_curves = {"RHC": {}, "SA": {}, "GA": {}}

    for size in [30, 60, 90]:
        print(f"\n===== FourPeaks (size={size}) =====")

        # Create FourPeaks problem instance
        problem_fp = DiscreteOpt(
            length=size,
            fitness_fn=FourPeaks(t_pct=0.15),
            maximize=True,
            max_val=2
        )

        # 1) RHC
        df_rhc, curves_rhc, eval_rhc = search_discrete_rhc(problem_fp, n_iter_bayes=10)
        print("\nRHC top combos:\n", tabulate(df_rhc, headers="keys", tablefmt="github"))

        # 2) SA
        df_sa, curves_sa, eval_sa = search_discrete_sa(problem_fp, n_iter_bayes=10)
        print("\nSA top combos:\n", tabulate(df_sa, headers="keys", tablefmt="github"))

        # 3) GA
        df_ga, curves_ga, eval_ga = search_discrete_ga(problem_fp, n_iter_bayes=10)
        print("\nGA top combos:\n", tabulate(df_ga, headers="keys", tablefmt="github"))

        # Combine results into one CSV in the "FourPeaks" subfolder
        header = f"FourPeaks Size {size} Results"
        combined = pd.concat([df_rhc, df_sa, df_ga], axis=0)
        save_table(combined, fp_csv, header=header, category="FourPeaks")

        # Create fitness vs iteration plots if the curves exist
        if curves_rhc:
            plot_fitness_vs_iteration(curves_rhc, "FourPeaks", "RHC", size, maximize=True)
        if curves_sa:
            plot_fitness_vs_iteration(curves_sa, "FourPeaks", "SA", size, maximize=True)
        if curves_ga:
            plot_fitness_vs_iteration(curves_ga, "FourPeaks", "GA", size, maximize=True)

        # After plotting fitness vs iteration for RHC, SA, and GA:
        if curves_rhc:
            plot_fevals_vs_iteration(curves_rhc[0], "FourPeaks", "RHC", size)
        if curves_sa:
            plot_fevals_vs_iteration(curves_sa[0], "FourPeaks", "SA", size)
        if curves_ga:
            plot_fevals_vs_iteration(curves_ga[0], "FourPeaks", "GA", size)

        # Plot fitness vs time (and hence F-Evals vs time is included)
        plot_fitness_vs_time(eval_rhc, "RHC", "FourPeaks", maximize=True)
        plot_fitness_vs_time(eval_sa, "SA", "FourPeaks", maximize=True)
        plot_fitness_vs_time(eval_ga, "GA", "FourPeaks", maximize=True)

        # Record best fitness and average runtime
        four_peaks_fitness["RHC"][size] = df_rhc["Fitness"].max()
        four_peaks_fitness["SA"][size] = df_sa["Fitness"].max()
        four_peaks_fitness["GA"][size] = df_ga["Fitness"].max()

        four_peaks_runtimes["RHC"][size] = np.mean([x['elapsed'] for x in eval_rhc])
        four_peaks_runtimes["SA"][size] = np.mean([x['elapsed'] for x in eval_sa])
        four_peaks_runtimes["GA"][size] = np.mean([x['elapsed'] for x in eval_ga])

        # Store representative curves for combined plotting
        if curves_rhc:
            four_peaks_curves["RHC"][size] = curves_rhc[0]
        if curves_sa:
            four_peaks_curves["SA"][size] = curves_sa[0]
        if curves_ga:
            four_peaks_curves["GA"][size] = curves_ga[0]

    # Summaries: Fitness vs Problem Size, Runtime vs Problem Size
    for algo in ["RHC", "SA", "GA"]:
        plot_fitness_vs_problem_size(four_peaks_fitness[algo], algo, maximize=True)
        sizes = sorted(four_peaks_runtimes[algo].keys())
        rts = [four_peaks_runtimes[algo][s] for s in sizes]
        plot_runtime_vs_problem_size(sizes, rts, algo)

    # Plot combined fitness vs iteration curves for FourPeaks
    for algo in ["RHC", "SA", "GA"]:
        if four_peaks_curves[algo]:
            plot_multiple_fitness_curves(four_peaks_curves[algo], "FourPeaks", algo, maximize=True)

    # =======================
    # TSP problem
    # =======================
    from mlrose_hiive import TSPOpt
    tsp_csv = "TSP_Results.csv"
    tsp_folder = os.path.join(CSV_DIR, "TSP")
    os.makedirs(tsp_folder, exist_ok=True)
    open(os.path.join(tsp_folder, tsp_csv), "w").close()

    # Create dictionary to store representative curves for TSP combined plotting
    tsp_curves = {"RHC": {}, "SA": {}, "GA": {}}

    for tsp_size in [10, 50, 100]:
        print(f"\n===== TSP (size={tsp_size}) =====")

        # Generate random coordinates for TSP
        coords = np.random.rand(tsp_size, 2) * 100.0
        coords = coords.tolist()

        # Now we define TSP problem as maximizing negative distance
        # effectively turning TSP into a "maximization" version
        problem_tsp = TSPOpt(length=tsp_size, coords=coords, maximize=True)

        # RHC
        df_rhc, curves_rhc, eval_rhc = search_discrete_rhc(problem_tsp, n_iter_bayes=10)
        print("\nRHC top combos:\n", tabulate(df_rhc, headers="keys", tablefmt="github"))

        # SA
        df_sa, curves_sa, eval_sa = search_discrete_sa(problem_tsp, n_iter_bayes=10)
        print("\nSA top combos:\n", tabulate(df_sa, headers="keys", tablefmt="github"))

        # GA
        df_ga, curves_ga, eval_ga = search_discrete_ga(problem_tsp, n_iter_bayes=10)
        print("\nGA top combos:\n", tabulate(df_ga, headers="keys", tablefmt="github"))

        header = f"TSP Size {tsp_size} Results"
        combined = pd.concat([df_rhc, df_sa, df_ga], axis=0)
        save_table(combined, tsp_csv, header=header, category="TSP")

        # Plot fitness vs iteration
        if curves_rhc:
            plot_fitness_vs_iteration(curves_rhc, "TSP", "RHC", tsp_size, maximize=True)
        if curves_sa:
            plot_fitness_vs_iteration(curves_sa, "TSP", "SA", tsp_size, maximize=True)
        if curves_ga:
            plot_fitness_vs_iteration(curves_ga, "TSP", "GA", tsp_size, maximize=True)

        # Plot fevals vs iteration for TSP
        if curves_rhc:
            plot_fevals_vs_iteration(curves_rhc[0], "TSP", "RHC", tsp_size)
        if curves_sa:
            plot_fevals_vs_iteration(curves_sa[0], "TSP", "SA", tsp_size)
        if curves_ga:
            plot_fevals_vs_iteration(curves_ga[0], "TSP", "GA", tsp_size)

        # Plot fitness vs time (and thus F-Evals vs time) for TSP
        plot_fitness_vs_time(eval_rhc, "RHC", "TSP", maximize=True)
        plot_fitness_vs_time(eval_sa, "SA", "TSP", maximize=True)
        plot_fitness_vs_time(eval_ga, "GA", "TSP", maximize=True)

        # Record best fitness & average runtime
        tsp_fitness["RHC"][tsp_size] = df_rhc["Fitness"].max()
        tsp_fitness["SA"][tsp_size] = df_sa["Fitness"].max()
        tsp_fitness["GA"][tsp_size] = df_ga["Fitness"].max()

        tsp_runtimes["RHC"][tsp_size] = np.mean([x['elapsed'] for x in eval_rhc])
        tsp_runtimes["SA"][tsp_size] = np.mean([x['elapsed'] for x in eval_sa])
        tsp_runtimes["GA"][tsp_size] = np.mean([x['elapsed'] for x in eval_ga])

        # Store representative curves for combined plotting
        if curves_rhc:
            tsp_curves["RHC"][tsp_size] = curves_rhc[0]
        if curves_sa:
            tsp_curves["SA"][tsp_size] = curves_sa[0]
        if curves_ga:
            tsp_curves["GA"][tsp_size] = curves_ga[0]

    # Summaries for TSP
    for algo in ["RHC", "SA", "GA"]:
        plot_fitness_vs_problem_size(tsp_fitness[algo], algo, maximize=True)
        sizes = sorted(tsp_runtimes[algo].keys())
        rts = [tsp_runtimes[algo][s] for s in sizes]
        plot_runtime_vs_problem_size(sizes, rts, algo)

    # Plot combined fitness vs iteration curves for TSP
    for algo in ["RHC", "SA", "GA"]:
        if tsp_curves[algo]:
            plot_multiple_fitness_curves(tsp_curves[algo], "TSP", algo, maximize=True)


###############################################################################
#  5) Master Orchestration for NN
#  -> Run Baseline, RHC, SA, GA on Marketing Data
###############################################################################
def run_nn_experiments():
    """
    1) Splits data into 70/15/15 for training, validation, test.
    2) Uses SMOTEENN on the training portion only to address imbalance.
    3) Ensures feature scaling (StandardScaler).
    4) Baseline: gradient_descent (alpha, learning_rate from FROZEN_NN_CONFIG).
    5) RHC, SA, GA: only activation + hidden_nodes from config;
       alpha & learning_rate are not used for random optimization methods.
    6) Summaries and best param combos for each approach.
    7) Encouragement:
       - "Ensure proper feature scaling" -> done via StandardScaler
       - "Tune hyperparameters using Bayesian optimization for longer" -> n_iter can be 10+
       - "Try different SA schedules" -> we have arith, geom, exp.
    """
    X_full, y_full = preprocess_marketing_campaign(MARKETING_CAMPAIGN_PATH)

    # Split 70/30
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=GLOBAL_SEED
    )
    # Then 15% each from that 30% => total 70/15/15
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=GLOBAL_SEED
    )

    # -----------------------------------------------------------------------------
    # Apply SMOTEENN for refined handling of imbalanced data on training portion
    # This helps create a balanced representation while removing borderline noise.
    # -----------------------------------------------------------------------------
    smote = SMOTEENN(
        smote=SMOTE(sampling_strategy="auto", k_neighbors=3),
        enn=EditedNearestNeighbours(n_neighbors=5),
        random_state=GLOBAL_SEED
    )

    X_train_res, y_train_res = smote.fit_resample(X_train_raw, y_train_raw)

    # -----------------------------------------------------------------------------
    # Scale the features. This is crucial for NNs and random search methods,
    # because large-scale features can hamper convergence or cause suboptimal search.
    # -----------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_val_arr = scaler.transform(X_val_raw)
    X_test_arr = scaler.transform(X_test_raw)

    # Convert to np arrays
    y_train_res = np.array(y_train_res)
    y_val_arr = np.array(y_val_raw)
    y_test_arr = np.array(y_test_raw)

    # =======================
    # 1) Baseline GD
    # =======================
    print("\n=== FROZEN NN CONFIGURATION (Section 3.2) ===")
    print(FROZEN_NN_CONFIG)
    baseline_summary, cv_train_scores = run_nn_frozen_baseline(
        X_train_res, y_train_res, X_val_arr, y_val_arr, X_test_arr, y_test_arr
    )
    print("\n=== NN Frozen (Baseline: gradient_descent) Results ===")
    print(tabulate([baseline_summary], headers="keys", tablefmt="github"))

    # Plot baseline train vs test performance
    test_score_dummy = [baseline_summary["Test_F1 (Holdout)"]] * len(cv_train_scores)
    plot_train_vs_test_performance(cv_train_scores, test_score_dummy, "Baseline_NN")

    # Save baseline row to CSV in 'NeuralNetworks' subfolder
    nn_csv = "NN_Results.csv"
    nn_folder = os.path.join(CSV_DIR, "NeuralNetworks")
    os.makedirs(nn_folder, exist_ok=True)
    open(os.path.join(nn_folder, nn_csv), "w").close()
    save_table(pd.DataFrame([baseline_summary]),
               nn_csv,
               header="Baseline NN Results (Gradient Descent)",
               category="NeuralNetworks")

    # =======================
    # 2) RHC
    # =======================
    df_rhc, eval_times_rhc = search_nn_rhc(X_train_res, y_train_res, X_test_arr, y_test_arr,
                                           n_iter_bayes=5)  ### UPDATED: unpack evaluation_times
    print("\n=== NN_RHC Tuning (Top 5 combos) ===")
    print(tabulate(df_rhc, headers="keys", tablefmt="github"))
    save_table(df_rhc, nn_csv, header="NN_RHC Tuning Results", category="NeuralNetworks")

    # Plot Fevals vs Time for NN RHC
    plot_fitness_vs_time(eval_times_rhc, "RHC", "NN", maximize=True)

    # =======================
    # 3) SA
    # =======================
    df_sa, eval_times_sa = search_nn_sa(X_train_res, y_train_res, X_test_arr, y_test_arr,
                                        n_iter_bayes=5)  ### UPDATED: unpack evaluation_times
    print("\n=== NN_SA Tuning (Top 5 combos) ===")
    print(tabulate(df_sa, headers="keys", tablefmt="github"))
    save_table(df_sa, nn_csv, header="NN_SA Tuning Results", category="NeuralNetworks")

    # Plot Fevals vs Time for NN SA
    plot_fitness_vs_time(eval_times_sa, "SA", "NN", maximize=True)

    # =======================
    # 4) GA
    # =======================
    df_ga, eval_times_ga = search_nn_ga(X_train_res, y_train_res, X_test_arr, y_test_arr,
                                        n_iter_bayes=5)  ### UPDATED: unpack evaluation_times
    print("\n=== NN_GA Tuning (Top 5 combos) ===")
    print(tabulate(df_ga, headers="keys", tablefmt="github"))
    save_table(df_ga, nn_csv, header="NN_GA Tuning Results", category="NeuralNetworks")

    # Plot Fevals vs Time for NN GA
    plot_fitness_vs_time(eval_times_ga, "GA", "NN", maximize=True)

    # Summaries
    summary_data = []

    # Baseline
    summary_data.append({
        "Algorithm": "Baseline (GD)",
        "CV_F1": baseline_summary["CV_F1"],
        "Test_F1": baseline_summary["Test_F1 (Holdout)"],
        "Test_F1_Macro": baseline_summary["Test_F1_Macro"],
        "Test_Balanced_Accuracy": baseline_summary["Test_Balanced_Accuracy"],
        "Elapsed_Time": baseline_summary["Elapsed_Time"]
    })

    # RHC Best
    best_rhc = df_rhc.loc[df_rhc["CV_F1"].idxmax()]
    summary_data.append({
        "Algorithm": "RHC-NN",
        "CV_F1": best_rhc["CV_F1"],
        "Test_F1": best_rhc["Test_F1 (Holdout)"],
        "Test_F1_Macro": best_rhc["Test_F1_Macro"],
        "Test_Balanced_Accuracy": best_rhc["Test_Balanced_Accuracy"],
        "Elapsed_Time": df_rhc["Elapsed_Time"].mean()
    })

    # SA Best
    best_sa = df_sa.loc[df_sa["CV_F1"].idxmax()]
    summary_data.append({
        "Algorithm": "SA-NN",
        "CV_F1": best_sa["CV_F1"],
        "Test_F1": best_sa["Test_F1 (Holdout)"],
        "Test_F1_Macro": best_sa["Test_F1_Macro"],
        "Test_Balanced_Accuracy": best_sa["Test_Balanced_Accuracy"],
        "Elapsed_Time": df_sa["Elapsed_Time"].mean()
    })

    # GA Best
    best_ga = df_ga.loc[df_ga["CV_F1"].idxmax()]
    summary_data.append({
        "Algorithm": "GA-NN",
        "CV_F1": best_ga["CV_F1"],
        "Test_F1": best_ga["Test_F1 (Holdout)"],
        "Test_F1_Macro": best_ga["Test_F1_Macro"],
        "Test_Balanced_Accuracy": best_ga["Test_Balanced_Accuracy"],
        "Elapsed_Time": df_ga["Elapsed_Time"].mean()
    })

    # Summarize results in single CSV (NN_Summary.csv)
    summary_path = os.path.join(nn_folder, "NN_Summary.csv")
    generate_summary_table(summary_data, summary_path)

    # --------------------------------------------------------
    # 5) Visualize Train vs Test for each best param set (RHC, SA, GA)
    # --------------------------------------------------------

    # RHC
    best_rhc_dict = {
        "algorithm": "random_hill_climb",
        "RHC_NumRestarts": int(best_rhc["RHC_NumRestarts"]),
        "RHC_MaxIterations": int(best_rhc["RHC_MaxIterations"]),
        "RHC_MaxAttempts": int(best_rhc["RHC_MaxAttempts"])
    }
    rhc_train, rhc_test = cross_validate_nn(X_train_res, y_train_res, best_rhc_dict)
    plot_train_vs_test_performance(rhc_train, rhc_test, "RHC-NN")

    # Re-run best NN RHC to obtain fitness curve for Fevals vs Iteration
    model_rhc = mlrose.NeuralNetwork(
        activation=FROZEN_NN_CONFIG["params_activation"],
        hidden_nodes=FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        early_stopping=True,
        clip_max=5.0,
        bias=True,
        is_classifier=True,
        curve=True,
        random_state=GLOBAL_SEED,
        algorithm="random_hill_climb",
        restarts=best_rhc_dict["RHC_NumRestarts"],
        max_iters=best_rhc_dict["RHC_MaxIterations"],
        max_attempts=best_rhc_dict["RHC_MaxAttempts"]
    )
    model_rhc.fit(X_train_res, y_train_res)
    if hasattr(model_rhc, 'fitness_curve') and model_rhc.fitness_curve is not None:
        plot_fevals_vs_iteration(model_rhc.fitness_curve, "NN", "RHC", "NN")

    # SA
    best_sa_dict = {
        "algorithm": "simulated_annealing",
        "SA_ScheduleType": best_sa["SA_ScheduleType"],
        "SA_InitialTemperature": float(best_sa["SA_InitialTemperature"]),
        "SA_MaxIterations": int(best_sa["SA_MaxIterations"]),
        "SA_MaxAttempts": int(best_sa["SA_MaxAttempts"]),
        "SA_TemperatureDecay": float(best_sa["SA_TemperatureDecay"])
    }
    sa_train, sa_test = cross_validate_nn(X_train_res, y_train_res, best_sa_dict)
    plot_train_vs_test_performance(sa_train, sa_test, "SA-NN")

    # Re-run best NN SA to obtain fitness curve for Fevals vs Iteration
    model_sa = mlrose.NeuralNetwork(
        activation=FROZEN_NN_CONFIG["params_activation"],
        hidden_nodes=FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        early_stopping=True,
        clip_max=5.0,
        bias=True,
        is_classifier=True,
        curve=True,
        random_state=GLOBAL_SEED,
        algorithm="simulated_annealing",
        schedule=schedule_object(
            0 if best_sa_dict["SA_ScheduleType"] == "exp" else (1 if best_sa_dict["SA_ScheduleType"] == "geom" else 2),
            best_sa_dict["SA_InitialTemperature"],
            best_sa_dict["SA_TemperatureDecay"]
        ),
        max_iters=best_sa_dict["SA_MaxIterations"],
        max_attempts=best_sa_dict["SA_MaxAttempts"]
    )
    model_sa.fit(X_train_res, y_train_res)
    if hasattr(model_sa, 'fitness_curve') and model_sa.fitness_curve is not None:
        plot_fevals_vs_iteration(model_sa.fitness_curve, "NN", "SA", "NN")

    # GA
    best_ga_dict = {
        "algorithm": "genetic_alg",
        "GA_PopulationSize": int(best_ga["GA_PopulationSize"]),
        "GA_MutationProb": float(best_ga["GA_MutationProb"]),
        "GA_CrossoverProb": float(best_ga["GA_CrossoverProb"]),
        "GA_MaxIterations": int(best_ga["GA_MaxIterations"]),
        "GA_MaxAttempts": int(best_ga["GA_MaxAttempts"])
    }
    ga_train, ga_test = cross_validate_nn(X_train_res, y_train_res, best_ga_dict)
    plot_train_vs_test_performance(ga_train, ga_test, "GA-NN")

    # Re-run best NN GA to obtain fitness curve for Fevals vs Iteration
    model_ga = mlrose.NeuralNetwork(
        activation=FROZEN_NN_CONFIG["params_activation"],
        hidden_nodes=FROZEN_NN_CONFIG["params_hidden_layer_sizes"],
        early_stopping=True,
        clip_max=5.0,
        bias=True,
        is_classifier=True,
        curve=True,
        random_state=GLOBAL_SEED,
        algorithm="genetic_alg",
        pop_size=best_ga_dict["GA_PopulationSize"],
        mutation_prob=best_ga_dict["GA_MutationProb"],
        max_iters=best_ga_dict["GA_MaxIterations"],
        max_attempts=best_ga_dict["GA_MaxAttempts"]
    )
    model_ga.fit(X_train_res, y_train_res)
    if hasattr(model_ga, 'fitness_curve') and model_ga.fitness_curve is not None:
        plot_fevals_vs_iteration(model_ga.fitness_curve, "NN", "GA", "NN")

    print("\n=== Done with Section 3.2. ===")


###############################################################################
#  6) Main
#  -> Orchestrates the above code
###############################################################################
def main():
    """
    Main pipeline that:
      1) Sets a global random seed for reproducibility.
      2) Runs discrete optimization experiments (RHC, SA, GA) on FourPeaks & TSP.
      3) Runs NN experiments (Baseline GD vs. RHC, SA, GA).
      4) Prints total runtime and where files are saved.
    """
    start_time = time.time()
    set_global_seed(GLOBAL_SEED)

    print("\n=== Section 3.1: Discrete Optimization Experiments ===")
    run_ro_optimization_problems()

    print("\n=== Section 3.2: Neural Network Weight Optimization Experiments ===")
    run_nn_experiments()

    total_time = time.time() - start_time
    print(f"\nTotal Program Runtime: {total_time:.4f} seconds")
    print(f"All files were saved to the '{BASE_OUTPUT_DIR}' folder.")
    print("End of Program.")


if __name__ == "__main__":
    main()
