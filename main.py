import os
import time
import argparse
from src.data_collector import build_dataset_from_repo
from src.preprocess import preprocess_dataset
from src.genetic_algorithm import FeatureSelectionGA

def main():
    # --- ARGUMENT PARSER SETUP ---
    parser = argparse.ArgumentParser(description="Auto-Maintainability Neuro-Genetic Framework")
    
    # Target and Data Arguments
    parser.add_argument("--repo", type=str, default="test_repos/flask", help="Path to the target repository")
    parser.add_argument("--raw-file", type=str, default="flask_dataset.csv", help="Output file for raw mined data")
    parser.add_argument("--processed-file", type=str, default="flask_dataset_processed.csv", help="Output file for scaled data")
    
    # Genetic Algorithm Hyperparameters
    parser.add_argument("--pop-size", type=int, default=15, help="Size of the GA population")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations to evolve")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Probability of a gene mutating (0.0 to 1.0)")
    
    # Fitness Weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Accuracy (minimizing MSE)")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for Parsimony (minimizing feature count)")
    
    # Pipeline Toggles
    parser.add_argument("--force-collect", action="store_true", help="Force re-mining the Git repository")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the raw data")

    args = parser.parse_args()

    print("="*60)
    print(" AUTO-MAINTAINABILITY NEURO-GENETIC FRAMEWORK")
    print("="*60)
    print(f"Config: Pop: {args.pop_size} | Gens: {args.generations} | Alpha: {args.alpha} | Beta: {args.beta}\n")

    # STEP 1: Mine Git & Code Syntax
    if args.force_collect or not os.path.exists(args.raw_file):
        print(f"[STEP 1] Mining Repository for Code & Evolutionary Metrics: {args.repo}")
        build_dataset_from_repo(args.repo, args.raw_file)
    else:
        print(f"[STEP 1] Skipping Collection. Using existing '{args.raw_file}'...")

    # STEP 2: Clean & Normalize
    if args.force_process or not os.path.exists(args.processed_file):
        print("\n[STEP 2] Preprocessing and Scaling Data...")
        preprocess_dataset(args.raw_file, args.processed_file)
    else:
        print(f"\n[STEP 2] Skipping Preprocessing. Using existing '{args.processed_file}'...")

    # STEP 3: Neuro-Genetic Feature Selection
    print("\n[STEP 3] Initializing Hybrid Neuro-Genetic Framework...")
    start_time = time.time()
    
    ga = FeatureSelectionGA(
        csv_file=args.processed_file,
        population_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        alpha=args.alpha,
        beta=args.beta
    )
    
    best_chromosome = ga.evolve()
    
    end_time = time.time()
    print(f"\n[FINISHED] Total Optimization Time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
