"""Quick test: Verify frog predator is hunting and moving correctly"""

from Frog_predator_neuro.simulation import Simulation


def test_frog_predator():
    """Launch frog simulation and check hunting behavior"""
    print("=== Frog Predator Neuro - Quick Test ===")
    print("Testing frog hunting behavior for 20 seconds...\n")
    
    sim = Simulation(headless=True)
    agent = sim.agents[0] if sim.agents else None
    
    if not agent:
        print("❌ ERROR: No agents spawned!")
        return False
    
    # Run 20 seconds
    for frame in range(2000):
        dt = 1.0 / 100.0
        sim.update(dt)
        if not sim.headless:
            sim.draw()
        
        # Check every 5 seconds
        if frame % 500 == 0 and frame > 0:
            time = frame / 100.0
            energy = agent.body_energy
            food_collected = agent.food_collected
            distance = agent.last_step_distance if hasattr(agent, 'last_step_distance') else 0
            x, y = agent.x, agent.y
            
            print(f"[{time:6.1f}s] Energy={energy:.3f} | Food={food_collected} | Pos=({x:7.1f}, {y:7.1f}) | Move={distance:.2f}")
    
    print("\n=== Results ===")
    print(f"Final energy: {agent.body_energy:.3f}")
    print(f"Food collected: {agent.food_collected}")
    
    # Validation
    if agent.food_collected > 0:
        print("\n✅ SUCCESS - Frog is hunting and eating!")
        return True
    elif agent.food_collected == 0 and agent.body_energy < 0.6:
        print("\n⚠️  PARTIAL - Frog is moving but not capturing prey")
        return False
    else:
        print("\n❌ FAIL - Frog not moving or hunting")
        return False


if __name__ == "__main__":
    success = test_frog_predator()
    exit(0 if success else 1)
