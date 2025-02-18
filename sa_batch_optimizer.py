import random
import math
from collections import defaultdict
from qubots.base_optimizer import BaseOptimizer

class SABatchOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimizer for the batch scheduling problem.

    Representation:
      - For each resource, a list (ordering) of tasks assigned to that resource.
      - A greedy batching procedure (in construct_batch_schedule) groups tasks with the same type,
        up to the resource’s capacity, and assigns start/finish times.
    
    Evaluation:
      - The candidate schedule is built from the ordering and then evaluated using the
        problem's own evaluate_solution method.
    
    Neighborhood Moves:
      - For a randomly chosen resource, either swap two tasks or reinsert a task at a new position.
    """

    def __init__(self, initial_temperature=1000, final_temperature=1, cooling_rate=0.95, iterations_per_temp=100):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Create an initial ordering solution: for each resource, list its tasks in a random order.
        resources = set(problem.resources)
        current_solution = {r: [] for r in resources}
        for t in range(problem.nb_tasks):
            r = problem.resources[t]
            current_solution[r].append(t)
        # Randomize order for each resource.
        for r in current_solution:
            random.shuffle(current_solution[r])
        
        current_obj, _ = self.evaluate_solution(current_solution, problem)
        best_solution = current_solution
        best_obj = current_obj

        T = self.initial_temperature
        
        while T > self.final_temperature:
            for _ in range(self.iterations_per_temp):
                new_solution = self.neighbor_solution(current_solution, problem)
                new_obj, _ = self.evaluate_solution(new_solution, problem)
                delta = new_obj - current_obj
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = new_solution
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_solution = current_solution
                        best_obj = current_obj
            T *= self.cooling_rate
        
        # Build the final schedule from the best ordering found.
        batch_schedule = self.construct_batch_schedule(best_solution, problem)
        return {'batch_schedule': batch_schedule}, best_obj

    def evaluate_solution(self, ordering_solution, problem):
        """
        Build the candidate schedule from the ordering and evaluate it using the problem's method.
        The candidate solution is a dictionary with key "batch_schedule".
        Returns a tuple (objective_value, None). The second element is not used.
        """
        schedule = self.construct_batch_schedule(ordering_solution, problem)
        obj_value = problem.evaluate_solution({"batch_schedule": schedule})
        return obj_value, None

    def neighbor_solution(self, solution, problem):
        """
        Generate a neighboring solution by modifying the ordering for a randomly chosen resource.
        Two moves are implemented: 
          - Swapping two tasks.
          - Removing one task and reinserting it at a different position.
        """
        new_solution = {r: list(tasks) for r, tasks in solution.items()}
        r = random.choice(list(new_solution.keys()))
        tasks = new_solution[r]
        if len(tasks) < 2:
            return new_solution
        
        if random.random() < 0.5:
            # Swap two tasks.
            i, j = random.sample(range(len(tasks)), 2)
            tasks[i], tasks[j] = tasks[j], tasks[i]
        else:
            # Reinsert a task at a new position.
            i = random.randrange(len(tasks))
            task = tasks.pop(i)
            j = random.randrange(len(tasks) + 1)
            tasks.insert(j, task)
        new_solution[r] = tasks
        return new_solution

    def construct_batch_schedule(self, solution, problem):
        """
        Reconstruct the batch schedule from the solution ordering.
        Returns a list of batches, where each batch is a dictionary with:
            'resource': resource id,
            'tasks': list of task indices in the batch,
            'start': batch start time,
            'end': batch finish time.
        The batching procedure groups consecutive tasks (with the same type) on a resource,
        up to the resource capacity.
        """
        batch_schedule = []
        for r, task_list in solution.items():
            current_time = 0
            i = 0
            while i < len(task_list):
                batch_type = problem.types[task_list[i]]
                batch = []
                # Group tasks into a batch (same type, up to capacity).
                while i < len(task_list) and len(batch) < problem.capacity[r] and \
                      problem.types[task_list[i]] == batch_type:
                    batch.append(task_list[i])
                    i += 1
                # Schedule the batch starting at current_time.
                batch_start = current_time
                batch_duration = max(problem.duration[t] for t in batch) if batch else 0
                batch_finish = batch_start + batch_duration
                batch_schedule.append({
                    'resource': r,
                    'tasks': batch,
                    'start': batch_start,
                    'end': batch_finish
                })
                current_time = batch_finish
        # Sort batches by start time.
        batch_schedule.sort(key=lambda b: b['start'])
        return batch_schedule
