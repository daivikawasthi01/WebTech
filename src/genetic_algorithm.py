"""
genetic_algorithm.py — Production version.

Changes from cloud version:
  - epochs: 8 → 100 (was reduced for Streamlit Cloud CPU constraints)
  - use_kfold: False → True during fitness evaluation for reliable MSE signal
    (single split on small datasets is noisy; k-fold averages out variance)
  - population_size default: 8 → 20
  - generations default: 5 → 30
  - stagnation_limit default: 3 → 8
  - min_mutation_rate: 0.02 → 0.03
  - n_epochs added as __init__ parameter so CLI can override without code changes
  - Interim ga_results.json write after every generation retained — still
    useful locally for monitoring long runs
  - Checkpoint/resume, elitism, memoization, adaptive mutation all retained
"""

import json
import os
import random
import pandas as pd
from src.ann_model import train_and_evaluate_ann


class FeatureSelectionGA:
    def __init__(
        self,
        csv_file: str,
        population_size: int     = 20,
        generations: int         = 30,
        mutation_rate: float     = 0.20,
        min_mutation_rate: float = 0.03,
        alpha: float             = 1.0,
        beta: float              = 0.5,
        stagnation_limit: int    = 8,
        log_transform: bool      = True,
        checkpoint_path: str     = "data/results/ga_checkpoint.json",
        n_epochs: int            = 100,   # ANN epochs per fitness evaluation
        use_kfold: bool          = True,  # k-fold for reliable MSE during evolution
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
        self.n_epochs          = n_epochs
        self.use_kfold         = use_kfold

        df = pd.read_csv(csv_file)
        self.num_features  = len(df.columns) - 2
        self.feature_names = df.columns[1:-1].tolist()

        self.evaluation_cache: dict = {}
        self.history: list          = []

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
        if self.generations <= 1:
            return self.min_mutation_rate
        rate = (self.min_mutation_rate
                + (self.mutation_rate - self.min_mutation_rate)
                * (1.0 - generation / self.generations))
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
        """
        Returns (fitness, mse). Memoised — identical chromosomes never re-trained.

        n_epochs=100: enough for the ANN to converge and give a reliable MSE
        signal to the GA. The cloud version used 8 epochs which was insufficient
        for meaningful gradient descent and produced noisy fitness rankings.

        use_kfold=True: stratified k-fold cross-validation averages out the
        variance from a single train/val split. On small datasets (50-200 files)
        a single split can vary by ±30% depending on which files land in the
        validation set. K-fold gives a much more stable fitness signal.
        """
        if chromosome in self.evaluation_cache:
            return self.evaluation_cache[chromosome]

        mse = train_and_evaluate_ann(
            self.csv_file,
            feature_mask  = list(chromosome),
            epochs        = self.n_epochs,
            batch_size    = 32,
            use_kfold     = self.use_kfold,
            log_transform = self.log_transform,
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
        tournament = random.sample(scored, k)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def crossover(self, p1: tuple, p2: tuple) -> tuple:
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
        print("=" * 60)
        print(" NEURO-GENETIC FEATURE SELECTION")
        print("=" * 60)
        print(f" Dataset    : {self.csv_file}")
        print(f" Features   : {self.num_features}")
        print(f" Population : {self.population_size} | Max gens: {self.generations}")
        print(f" Mutation   : {self.mutation_rate:.2f} → {self.min_mutation_rate:.2f} (adaptive)")
        print(f" Alpha: {self.alpha}  Beta: {self.beta}  Stagnation: {self.stagnation_limit}")
        print(f" ANN epochs : {self.n_epochs} | K-fold: {self.use_kfold}")
        print("=" * 60 + "\n")

        self.history      = []
        population        = [self.generate_random_chromosome()
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
                cached       = chrom in self.evaluation_cache
                fitness, mse = self.calculate_fitness(chrom)
                scored.append((chrom, fitness, mse))
                label = "(cached)" if cached else ""
                print(f"  [{i+1:02d}] feats: {sum(chrom):02d}/{self.num_features}"
                      f"  MSE: {mse:.4f}  fit: {fitness:.4f} {label}")

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

            # Write interim results after every generation for monitoring
            if best_chrom_ever is not None:
                _interim_names = [self.feature_names[i]
                                  for i, v in enumerate(best_chrom_ever) if v == 1]
                _interim = {
                    'chromosome':    list(best_chrom_ever),
                    'feature_names': _interim_names,
                    'best_mse':      float(best_mse_ever),
                    'best_fitness':  float(best_fitness_ever),
                    'n_selected':    int(sum(best_chrom_ever)),
                    'n_total':       self.num_features,
                    'history':       self.history,
                    'cache_size':    len(self.evaluation_cache),
                    'complete':      False,
                }
                try:
                    with open("data/results/ga_results.json", 'w') as _f:
                        import json as _json
                        _json.dump(_interim, _f, indent=2)
                except Exception:
                    pass

            if stagnation >= self.stagnation_limit:
                print(f"  [CONVERGED] No improvement for {self.stagnation_limit} "
                      f"generations.\n")
                break

            # Build next generation
            scored.sort(key=lambda x: x[1], reverse=True)
            next_gen = [item[0] for item in scored[:2]]   # elitism: top 2
            while len(next_gen) < self.population_size:
                p1 = self.tournament_selection(scored)
                p2 = self.tournament_selection(scored)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutate(c1, gen))
                if len(next_gen) < self.population_size:
                    next_gen.append(self.mutate(c2, gen))
            population = next_gen

        selected_names = [self.feature_names[i]
                          for i, v in enumerate(best_chrom_ever) if v == 1]

        print("=" * 60)
        print(" OPTIMISATION COMPLETE")
        print(f" Best MSE    : {best_mse_ever:.4f}")
        print(f" Best Fitness: {best_fitness_ever:.4f}")
        print(f" Features    : {sum(best_chrom_ever)}/{self.num_features}")
        print(f" Cache hits  : {len(self.evaluation_cache)} unique evaluations")
        print("\n Selected features:")
        for name in selected_names:
            print(f"   + {name}")
        print("=" * 60)

        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        result = {
            'chromosome':    list(best_chrom_ever),
            'feature_names': selected_names,
            'best_mse':      float(best_mse_ever),
            'best_fitness':  float(best_fitness_ever),
            'n_selected':    int(sum(best_chrom_ever)),
            'n_total':       self.num_features,
            'history':       self.history,
            'cache_size':    len(self.evaluation_cache),
            'complete':      True,
        }
        return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ga = FeatureSelectionGA(
        csv_file        = "data/combined_dataset_clean.csv",
        population_size = 20,
        generations     = 30,
        mutation_rate   = 0.20,
        min_mutation_rate = 0.03,
        alpha           = 1.0,
        beta            = 0.5,
        stagnation_limit = 8,
        n_epochs        = 100,
        use_kfold       = True,
    )
    results = ga.evolve()