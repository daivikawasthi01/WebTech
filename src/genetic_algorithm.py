import random
import numpy as np
import pandas as pd
from src.ann_model import train_and_evaluate_ann

class FeatureSelectionGA:
    def __init__(self, csv_file, population_size=10, generations=5, mutation_rate=0.1, alpha=1.0, beta=0.5):
        self.csv_file = csv_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Determine number of features directly from the dataset (excluding file_name and target)
        df = pd.read_csv(csv_file)
        self.num_features = len(df.columns) - 2
        self.feature_names = df.columns[1:-1].tolist()
        
        # Fitness weights
        self.alpha = alpha # Weight for accuracy (MSE)
        self.beta = beta   # Weight for parsimony (having fewer features)
        
        # Caching/Memoization: To avoid re-training the ANN for previously seen chromosomes
        self.evaluation_cache = {}

    def generate_random_chromosome(self):
        """Creates a random array of 1s and 0s as a Tuple (for hashing)."""
        while True:
            chrom = tuple(random.choice([0, 1]) for _ in range(self.num_features))
            if sum(chrom) > 0: # Ensure at least one feature is selected
                return chrom

    def calculate_fitness(self, chromosome):
        """
        Evaluates the chromosome using the PyTorch ANN.
        Caches results to save massive computational time on repeats.
        """
        # If we have evaluated this exact feature mask before, pull it from memory!
        if chromosome in self.evaluation_cache:
            return self.evaluation_cache[chromosome]
            
        # 1. Run the ANN and get the Validation MSE
        mse = train_and_evaluate_ann(self.csv_file, feature_mask=list(chromosome), epochs=30, batch_size=16)
        
        # 2. Extract elements for formula
        n_selected = sum(chromosome)
        n_total = self.num_features
        
        # 3. Calculate Fitness using the exact Proposal Formula
        # Fitness = alpha * (1/MSE) + beta * (1 - N_selected/N_total)
        accuracy_term = self.alpha * (1.0 / (mse + 1e-6))
        parsimony_term = self.beta * (1.0 - (n_selected / n_total))
        
        fitness = accuracy_term + parsimony_term
        
        # Save to cache before returning
        self.evaluation_cache[chromosome] = (fitness, mse)
        return fitness, mse

    def tournament_selection(self, scored_population, k=3):
        """Tournament selection: picks k random individuals, returns the best. Better for diversity!"""
        tournament = random.sample(scored_population, k)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0] # Return the chromosome of the winner

    def crossover(self, parent1, parent2):
        """Single-point crossover to breed two parents."""
        crossover_point = random.randint(1, self.num_features - 1)
        child1 = tuple(parent1[:crossover_point] + parent2[crossover_point:])
        child2 = tuple(parent2[:crossover_point] + parent1[crossover_point:])
        
        # Failsafe for all zeros
        child1 = self._ensure_valid_chromosome(child1)
        child2 = self._ensure_valid_chromosome(child2)
        return child1, child2

    def mutate(self, chromosome):
        """Randomly flips bits based on the mutation rate."""
        mutated = list(chromosome)
        for i in range(self.num_features):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 if mutated[i] == 0 else 0
        return self._ensure_valid_chromosome(tuple(mutated))

    def _ensure_valid_chromosome(self, chromosome):
        """Helper to ensure we never return an all-zero chromosome."""
        if sum(chromosome) == 0:
            lst = list(chromosome)
            lst[random.randint(0, self.num_features - 1)] = 1
            return tuple(lst)
        return chromosome

    def evolve(self):
        print(f"--- Starting Neuro-Genetic Optimization ---")
        print(f"Dataset: {self.csv_file} | Total Features: {self.num_features}")
        print(f"Population: {self.population_size} | Generations: {self.generations}\n")
        
        # Initialize Population
        population = [self.generate_random_chromosome() for _ in range(self.population_size)]
        
        best_chromosome_ever = None
        best_fitness_ever = -float('inf')
        best_mse_ever = float('inf')

        for generation in range(self.generations):
            print(f"--- Generation {generation + 1}/{self.generations} ---")
            
            # Evaluate Population
            scored_population = []
            for i, chrom in enumerate(population):
                fitness, mse = self.calculate_fitness(chrom)
                scored_population.append((chrom, fitness, mse))
                cached = "(Cached)" if chrom in self.evaluation_cache else ""
                print(f" Ind {i+1:02d} | Features: {sum(chrom):02d}/{self.num_features} | MSE: {mse:.4f} | Fitness: {fitness:.4f} {cached}")
                
                # Track Global Best
                if fitness > best_fitness_ever:
                    best_fitness_ever = fitness
                    best_chromosome_ever = chrom
                    best_mse_ever = mse
                    
            # Sort by fitness (Highest fitness is first)
            scored_population.sort(key=lambda x: x[1], reverse=True)
            
            # Next Generation via Elitism, Tournament Selection, Crossover, and Mutation
            next_generation = []
            
            # Elitism: Keep top 2 unaltered
            next_generation.extend([item[0] for item in scored_population[:2]])
            
            while len(next_generation) < self.population_size:
                p1 = self.tournament_selection(scored_population)
                p2 = self.tournament_selection(scored_population)
                
                c1, c2 = self.crossover(p1, p2)
                next_generation.append(self.mutate(c1))
                if len(next_generation) < self.population_size:
                    next_generation.append(self.mutate(c2))
                    
            population = next_generation
            print(f" Best of Gen: Features {sum(scored_population[0][0])} | MSE {scored_population[0][2]:.4f} | Fit {scored_population[0][1]:.4f}\n")

        print("=========================================")
        print("Optimization Complete!")
        print(f"Best Target Validation MSE: {best_mse_ever:.4f}")
        print(f"Global Best Fitness Score: {best_fitness_ever:.4f}")
        print(f"Optimal Feature Mask: {best_chromosome_ever}")
        print("Selected Features:")
        for i, val in enumerate(best_chromosome_ever):
            if val == 1:
                print(f" - {self.feature_names[i]}")
        print("=========================================")
        return best_chromosome_ever

if __name__ == "__main__":
    ga = FeatureSelectionGA(
        csv_file="flask_dataset_processed.csv",
        population_size=10, 
        generations=4, 
        mutation_rate=0.15,
        alpha=1.0,  # Accuracy heavily weighted
        beta=0.5    # Parsimony slightly less weighted
    )
    best_features = ga.evolve()
