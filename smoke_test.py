from frog_lib import BioFlyCatchingSimulation

if __name__ == '__main__':
    sim = BioFlyCatchingSimulation(width=200, height=150, num_flies=2, headless=True)
    sim.run_simulation(50)
    print('SMOKE_DONE')
    print(sim.get_statistics())
