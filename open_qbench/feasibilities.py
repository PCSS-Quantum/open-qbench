from qlauncher.problems import JSSP


def JSSPFeasibility(sample: str, problem: JSSP) -> bool:
    """Check whether a given sample produces a feasible solution for a given jssp problem instance"""
    timesteps = problem.max_time
    machines = {}
    chars = [x for x in sample[::-1]]
    for i, product in enumerate(problem.instance.values()):
        timestep = 0
        for machine_step in product:
            name, time = machine_step

            if name not in machines:
                machines[name] = [None] * timesteps

            while timestep < timesteps and len(chars) > 0 and int(chars.pop()) != 1:
                timestep += 1
                continue

            for _ in range(time):
                # Tried to schedule outside of bounds
                if timestep >= timesteps:
                    return False
                # Overlapping schedules
                if machines[name][timestep] is not None:
                    return False

                machines[name][timestep] = i
                timestep += 1
    return True
