"""
genetic_algorithm.py

Changes:
1. Adaptive mutation rate: exponential decay from mutation_rate -> min_mutation_rate.
   (Previously the formula was linear despite the docstring claiming exponential;
    and the 'decay' variable was computed but never referenced. Both are now fixed.)
2. Generation history for Streamlit convergence plotting.
3. evolve() returns a full results dict.
4. tournament_selection k_safe guard: random.sample(scored, k) raises ValueError
   when len(scored) < k (e.g. --pop-size 1 or 2). Fixed with min(k, len(scored)).
5. crossover single-feature guard: randint(1, num_features-1) raises ValueError
   when num_features == 1. Returns parents unchanged in that case.
"""

import json
import math
import os
import random
import pandas as pd
from src.ann_model import train_and_evaluate_ann


class FeatureSelectionGA:
    def __init__(
        self,
        csv_file: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.20,
        min_mutation_rate: float = 0.02,
        alpha: float = 1.0,
        beta: float = 0.5,
        stagnation_limit: int = 5,
        log_transform: bool = True,
        checkpoint_path: str = "data/results/ga_checkpoint.json",
    ):
        self.csv_file          = csv_file
        self.population_size   = population_size
        self.generations       = generations
        self.mutation_rate     = mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.alpha             = alpha
        self.beta              = beta
        self.stagnation_limit  = stagnation_limit
        self.log_transform     = log_transform
        self.checkpoint_path   = checkpoint_path

        df = pd.read_csv(csv_file)
        self.num_features  = len(df.columns) - 2
        self.feature_names = df.columns[1:-1].tolist()

        self.evaluation_cache: dict = {}
        self.history: list = []

    # -----------------------------------------------------------------------
    # Checkpoint save / resume
    # -----------------------------------------------------------------------
    def _save_checkpoint(self, generation, population, best_chrom,
                         best_fitness, best_mse, stagnation):
        os.makedirs(os.path.dirname(self.checkpoint_path) or '.', exist_ok=True)
        ckpt = {
            'generation':   generation,
            'population':   [list(c) for c in population],
            'best_chrom':   list(best_chrom) if best_chrom else None,
            'best_fitness': float(best_fitness),
            'best_mse':     float(best_mse),
            'stagnation':   stagnation,
            'history':      self.history,
            'cache':        {str(k): list(v)
                             for k, v in self.evaluation_cache.items()},
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(ckpt, f)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path) as f:
                ckpt = json.load(f)
            restored_cache = {}
            for k_str, v in ckpt.get('cache', {}).items():
                try:
                    key = tuple(int(x) for x in
                                k_str.strip('()').replace(' ', '').split(',')
                                if x.strip())
                    restored_cache[key] = tuple(v)
                except Exception:
                    pass
            self.evaluation_cache = restored_cache
            self.history = ckpt.get('history', [])
            print(f"  [GA] Resuming from checkpoint — "
                  f"generation {ckpt['generation'] + 1}, "
                  f"cache size: {len(self.evaluation_cache)}")
            return ckpt
        except Exception as e:
            print(f"  [GA] Checkpoint load failed ({e}). Starting fresh.")
            return None

    # -----------------------------------------------------------------------
    # Adaptive mutation rate
    # -----------------------------------------------------------------------
    def _current_mutation_rate(self, generation: int) -> float:
        """
        Exponential decay from mutation_rate -> min_mutation_rate.
        Formula: rate = min + (max - min) * exp(-decay * gen)
        """
        if self.generations <= 1:
            return self.min_mutation_rate
        decay = 5.0 / self.generations
        rate  = (self.min_mutation_rate
                 + (self.mutation_rate - self.min_mutation_rate)
                 * math.exp(-decay * generation))
        return max(self.min_mutation_rate, rate)

    # -----------------------------------------------------------------------
    # Chromosome helpers
    # -----------------------------------------------------------------------
    def generate_random_chromosome(self) -> tuple:
        while True:
            chrom = tuple(random.choice([0, 1]) for _ in range(self.num_features))
            if sum(chrom) > 0:
                return chrom

    def _ensure_valid(self, chrom: tuple) -> tuple:
        if sum(chrom) == 0:
            lst = list(chrom)
            lst[random.randint(0, self.num_features - 1)] = 1
            return tuple(lst)
        return chrom

    # -----------------------------------------------------------------------
    # Fitness
    # -----------------------------------------------------------------------
    def calculate_fitness(self, chromosome: tuple) -> tuple:
        if chromosome in self.evaluation_cache:
            return self.evaluation_cache[chromosome]
        mse = train_and_evaluate_ann(
            self.csv_file,
            feature_mask=list(chromosome),
            epochs=50,
            batch_size=16,
            use_kfold=False,
            log_transform=self.log_transform,
        )
        n_selected = sum(chromosome)
        fitness = (self.alpha * (1.0 / (mse + 1e-6))
                   + self.beta * (1.0 - n_selected / self.num_features))
        self.evaluation_cache[chromosome] = (fitness, mse)
        return fitness, mse

    # -----------------------------------------------------------------------
    # Selection, crossover, mutation
    # -----------------------------------------------------------------------
    def tournament_selection(self, scored: list, k: int = 3) -> tuple:
        # FIX: random.sample(scored, k) raises ValueError when len(scored) < k
        # (e.g. --pop-size 1 or --pop-size 2). Clamp k to available pool size.
        k_safe     = min(k, len(scored))
        tournament = random.sample(scored, k_safe)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def crossover(self, p1: tuple, p2: tuple) -> tuple:
        # FIX: randint(1, self.num_features - 1) raises ValueError when
        # num_features == 1 because the range is empty (randint(1, 0)).
        # Return parents unchanged — crossover has no meaningful effect on
        # length-1 chromosomes anyway.
        if self.num_features <= 1:
            return p1, p2
        point = random.randint(1, self.num_features - 1)
        c1 = self._ensure_valid(p1[:point] + p2[point:])
        c2 = self._ensure_valid(p2[:point] + p1[point:])
        return c1, c2

    def mutate(self, chrom: tuple, generation: int) -> tuple:
        rate    = self._current_mutation_rate(generation)
        mutated = [1 - b if random.random() < rate else b for b in chrom]
        return self._ensure_valid(tuple(mutated))

    # -----------------------------------------------------------------------
    # Main evolution loop
    # -----------------------------------------------------------------------
    def evolve(self) -> dict:
        print("=" * 55)
        print("  NEURO-GENETIC FEATURE SELECTION")
        print("=" * 55)
        print(f"  Dataset    : {self.csv_file}")
        print(f"  Features   : {self.num_features}")
        print(f"  Population : {self.population_size} | Max gens : {self.generations}")
        print(f"  Mutation   : {self.mutation_rate:.2f} -> {self.min_mutation_rate:.2f} (adaptive)")
        print(f"  Alpha: {self.alpha}  Beta: {self.beta}  Stagnation: {self.stagnation_limit}")
        print("=" * 55 + "\n")

        self.history = []
        population = [self.generate_random_chromosome()
                      for _ in range(self.population_size)]
        best_chrom_ever   = None
        best_fitness_ever = -float('inf')
        best_mse_ever     = float('inf')
        stagnation        = 0
        start_gen         = 0

        ckpt = self._load_checkpoint()
        if ckpt is not None:
            start_gen         = ckpt['generation'] + 1
            population        = [tuple(c) for c in ckpt['population']]
            best_chrom_ever   = tuple(ckpt['best_chrom']) if ckpt['best_chrom'] else None
            best_fitness_ever = ckpt['best_fitness']
            best_mse_ever     = ckpt['best_mse']
            stagnation        = ckpt['stagnation']

        for gen in range(start_gen, self.generations):
            current_rate = self._current_mutation_rate(gen)
            print(f"--- Generation {gen+1}/{self.generations} "
                  f"(mutation rate: {current_rate:.3f}) ---")

            scored = []
            for i, chrom in enumerate(population):
                cached = chrom in self.evaluation_cache
                fitness, mse = self.calculate_fitness(chrom)
                scored.append((chrom, fitness, mse))
                label = "(cached)" if cached else ""
                print(f"  [{i+1:02d}] feats: {sum(chrom):02d}/{self.num_features}"
                      f"  MSE: {mse:.4f}  fit: {fitness:.4f}  {label}")

            gen_best = max(scored, key=lambda x: x[1])
            if gen_best[1] > best_fitness_ever:
                best_fitness_ever = gen_best[1]
                best_chrom_ever   = gen_best[0]
                best_mse_ever     = gen_best[2]
                stagnation        = 0
            else:
                stagnation += 1

            self.history.append({
                'generation':      gen + 1,
                'best_mse':        float(gen_best[2]),
                'global_best_mse': float(best_mse_ever),
                'best_fitness':    float(gen_best[1]),
                'n_features':      int(sum(gen_best[0])),
                'mutation_rate':   float(current_rate),
            })

            print(f"  >> Gen best  — feats: {sum(gen_best[0])} | "
                  f"MSE: {gen_best[2]:.4f} | Fit: {gen_best[1]:.4f}")
            print(f"  >> Global best MSE: {best_mse_ever:.4f} "
                  f"stagnation: {stagnation}/{self.stagnation_limit}\n")

            self._save_checkpoint(gen, population, best_chrom_ever,
                                  best_fitness_ever, best_mse_ever, stagnation)

            if stagnation >= self.stagnation_limit:
                print(f"  [CONVERGED] No improvement for {self.stagnation_limit} gens.\n")
                break

            scored.sort(key=lambda x: x[1], reverse=True)
            next_gen = [item[0] for item in scored[:2]]  # elitism: top 2
            while len(next_gen) < self.population_size:
                p1, p2 = (self.tournament_selection(scored),
                          self.tournament_selection(scored))
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutate(c1, gen))
                if len(next_gen) < self.population_size:
                    next_gen.append(self.mutate(c2, gen))
            population = next_gen

        selected_names = [self.feature_names[i]
                          for i, v in enumerate(best_chrom_ever) if v == 1]

        print("=" * 55)
        print("  OPTIMIZATION COMPLETE")
        print(f"  Best MSE     : {best_mse_ever:.4f}")
        print(f"  Best Fitness : {best_fitness_ever:.4f}")
        print(f"  Features     : {sum(best_chrom_ever)}/{self.num_features}")
        print(f"  Cache hits   : {len(self.evaluation_cache)} unique evals")
        print("\n  Selected features:")
        for name in selected_names:
            print(f"    + {name}")
        print("=" * 55)

        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        return {
            'chromosome':    best_chrom_ever,
            'feature_names': selected_names,
            'best_mse':      float(best_mse_ever),
            'best_fitness':  float(best_fitness_ever),
            'n_selected':    int(sum(best_chrom_ever)),
            'n_total':       self.num_features,
            'history':       self.history,
            'cache_size':    len(self.evaluation_cache),
        }


if __name__ == "__main__":
    ga = FeatureSelectionGA(
        csv_file="data/flask_dataset.csv",
        population_size=10, generations=4,
        mutation_rate=0.20, min_mutation_rate=0.03,
        alpha=1.0, beta=0.5, stagnation_limit=3,
    )
    results = ga.evolve()