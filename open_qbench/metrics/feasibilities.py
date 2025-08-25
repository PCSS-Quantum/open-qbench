from itertools import chain

from qlauncher.problems import JSSP


def JSSPFeasibility(sample: str, problem: JSSP) -> bool:
    """
    Check whether a given sample produces a feasible solution for a given jssp problem instance,
    i.e every task is scheduled and no two tasks are overlapping
    """
    max_timesteps = problem.max_time
    machines = {}
    total_machine_steps = 0
    machine_steps_scheduled = 0
    for name, _ in chain(*problem.instance.values()):
        total_machine_steps += 1
        if name not in machines:
            machines[name] = [None] * max_timesteps

    chars = [x for x in sample[::-1]]
    for i, product in enumerate(problem.instance.values()):
        timestep = 0
        for machine_step in product:
            name, time = machine_step

            while timestep < max_timesteps and len(chars) > 0 and chars.pop() != "1":
                timestep += 1

            if timestep > max_timesteps - time:
                return False

            for _ in range(time):
                if machines[name][timestep] is not None:
                    return False
                machines[name][timestep] = i
                timestep += 1

            machine_steps_scheduled += 1

    return machine_steps_scheduled == total_machine_steps
