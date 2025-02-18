import math
import random
from copy import deepcopy
from qubots.base_optimizer import BaseOptimizer

class SimulatedAnnealingBatchScheduler(BaseOptimizer):
    """
    Simulated Annealing optimizer for Batch Scheduling Problems.
    
    Key Features:
      - Starts with solution where each task is in its own batch (from problem.random_solution())
      - Proposes neighbor solutions by:
        1. Merging compatible batches
        2. Splitting large batches
        3. Shifting batch start times
      - Uses exponential cooling schedule
      - Maintains solution feasibility through careful perturbation operators
    """
    
    def __init__(self, 
                 initial_temp=10000,
                 cooling_rate=0.95,
                 iterations_per_temp=100,
                 max_idle_shift=10):
        """
        Parameters:
          initial_temp: Starting temperature for SA
          cooling_rate: Temperature multiplier per iteration (0 < rate < 1)
          iterations_per_temp: Number of iterations at each temperature level
          max_idle_shift: Maximum allowed time shift for batch movement (minutes)
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.max_idle_shift = max_idle_shift

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Initialize solution
        current_sol = initial_solution if initial_solution else problem.random_solution()
        current_cost = problem.evaluate_solution(current_sol)
        
        best_sol = deepcopy(current_sol)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        while temp > 1e-5:  # Termination condition
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor_sol = self._generate_neighbor(problem, current_sol)
                neighbor_cost = problem.evaluate_solution(neighbor_sol)
                
                # Acceptance probability
                delta = neighbor_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_sol = neighbor_sol
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_sol = deepcopy(current_sol)
                        best_cost = current_cost
                        
            # Cool down
            temp *= self.cooling_rate
            
        return best_sol, best_cost

    def _generate_neighbor(self, problem, current_sol):
        """
        Generates a neighbor solution using three perturbation strategies:
        1. Batch merge (40% probability)
        2. Batch split (30% probability)
        3. Time shift (30% probability)
        """
        new_sol = deepcopy(current_sol)
        batches = new_sol["batch_schedule"]
        
        rand = random.random()
        
        if rand < 0.4 and len(batches) > 1:
            # Attempt batch merge
            self._merge_batches(problem, batches)
        elif rand < 0.7 and len(batches) > 0:
            # Attempt batch split
            self._split_batch(problem, batches)
        else:
            # Attempt time shift
            self._shift_batches(batches)
            
        return new_sol

    def _merge_batches(self, problem, batches):
        """Merges two compatible batches on the same resource"""
        # Group batches by resource and type
        resource_groups = {}
        for i, batch in enumerate(batches):
            key = (batch["resource"], problem.types[batch["tasks"][0]])
            resource_groups.setdefault(key, []).append(i)
        
        # Find merge candidates
        merge_candidates = [v for v in resource_groups.values() if len(v) >= 2]
        if not merge_candidates:
            return
        
        # Select random candidate group
        group = random.choice(merge_candidates)
        i1, i2 = random.sample(group, 2)
        b1 = batches[i1]
        b2 = batches[i2]
        
        # Check capacity
        combined_size = len(b1["tasks"]) + len(b2["tasks"])
        if combined_size > problem.capacity[b1["resource"]]:
            return
        
        # Merge batches
        merged_batch = {
            "resource": b1["resource"],
            "tasks": b1["tasks"] + b2["tasks"],
            "start": min(b1["start"], b2["start"]),
            "end": max(b1["end"], b2["end"])
        }
        
        # Remove original batches and add merged
        batches[:] = [b for j, b in enumerate(batches) if j not in {i1, i2}]
        batches.append(merged_batch)

    def _split_batch(self, problem, batches):
        """Splits a batch into two smaller batches"""
        # Find splittable batches (size >= 2)
        splittable = [i for i, b in enumerate(batches) if len(b["tasks"]) >= 2]
        if not splittable:
            return
        
        split_idx = random.choice(splittable)
        batch = batches[split_idx]
        
        # Split tasks into two groups
        split_point = random.randint(1, len(batch["tasks"])-1)
        tasks1 = batch["tasks"][:split_point]
        tasks2 = batch["tasks"][split_point:]
        
        # Create new batches
        new_batch1 = {
            "resource": batch["resource"],
            "tasks": tasks1,
            "start": batch["start"],
            "end": batch["start"] + problem.duration[tasks1[0]]
        }
        new_batch2 = {
            "resource": batch["resource"],
            "tasks": tasks2,
            "start": batch["start"],
            "end": batch["start"] + problem.duration[tasks2[0]]
        }
        
        # Replace original batch with split batches
        batches.pop(split_idx)
        batches.extend([new_batch1, new_batch2])

    def _shift_batches(self, batches):
        """Shifts batch times within resource constraints"""
        if not batches:
            return
        
        # Select random batch
        batch = random.choice(batches)
        resource = batch["resource"]
        
        # Find all batches on the same resource
        resource_batches = sorted([b for b in batches if b["resource"] == resource],
                                  key=lambda x: x["start"])
        
        # Find possible shift window
        idx = resource_batches.index(batch)
        prev_end = resource_batches[idx-1]["end"] if idx > 0 else 0
        next_start = resource_batches[idx+1]["start"] if idx < len(resource_batches)-1 else float('inf')
        
        max_shift = min(
            self.max_idle_shift,
            next_start - batch["end"],
            batch["start"] - prev_end
        )
        
        if max_shift <= 0:
            return
        
        shift = random.randint(-max_shift, max_shift)
        batch["start"] += shift
        batch["end"] += shift