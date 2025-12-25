#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioFrog v2.0 - Simple Console Launcher
This script provides a user-friendly way to run the BioFrog simulation
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress Unicode errors by using ascii-friendly output
import io
import codecs

# Try to set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import subprocess
    try:
        # Set console code page to UTF-8
        subprocess.run(['chcp', '65001'], capture_output=True)
    except:
        pass

# Import the simulation
try:
    from frog_lib import BioFlyCatchingSimulation, BioFrogAgent
    IMPORT_OK = True
except ImportError as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


def print_header():
    """Print a simple text header"""
    print("\n" + "="*70)
    print("  BioFrog v2.0 - Biological Neural Network Frog Simulator")
    print("  A realistic frog hunting simulation with AI learning")
    print("="*70 + "\n")


def option_1_full_simulation():
    """OPTION 1: Full simulation with visualization"""
    print("\n" + "-"*70)
    print("FULL SIMULATION WITH VISUALIZATION")
    print("-"*70)
    print("""
Parameters:
  - Screen size: 800x600 px
  - Flies: 20
  - Mode: Automatic (childhood -> adult)
  - Learning: Active (STDP + dopamine)
  - Duration: ~2-3 minutes
  
Controls:
  - SPACE: pause/resume
  - Q: quit
  - R: reset
    """)

    input("Press ENTER to start...")
    
    if not IMPORT_OK:
        print(f"\nERROR: Could not import frog_lib\n{IMPORT_ERROR}")
        return
    
    try:
        sim = BioFlyCatchingSimulation(
            width=800,
            height=600,
            num_flies=20,
            headless=False
        )
        print("\nStarting full simulation...\n")
        sim.run_simulation(5000)
        
        print("\nSimulation completed!")
        stats = sim.get_statistics()
        print("Flies caught: {}".format(stats.get('caught_flies', 0)))
        success_rate = stats.get('success_rate', 0) * 100
        print("Success rate: {:.1f}%".format(success_rate))
        
    except Exception as e:
        print("ERROR during simulation: {}".format(e))


def option_2_quick_test():
    """OPTION 2: Quick test without visualization"""
    print("\n" + "-"*70)
    print("QUICK TEST (NO VISUALIZATION)")
    print("-"*70)
    print("""
Parameters:
  - Flies: 10
  - Steps: 500 (fast!)
  - Visualization: Disabled
  - Duration: ~5 seconds
    """)

    input("Press ENTER to start...")
    
    if not IMPORT_OK:
        print("ERROR: Could not import frog_lib\n{}".format(IMPORT_ERROR))
        return
    
    try:
        sim = BioFlyCatchingSimulation(
            width=400,
            height=300,
            num_flies=10,
            headless=True
        )
        print("\nRunning test (this will take ~5 seconds)...\n")
        sim.run_simulation(500)
        
        print("\nTest completed!")
        stats = sim.get_statistics()
        print("Flies caught: {}".format(stats.get('caught_flies', 0)))
        success_rate = stats.get('success_rate', 0) * 100
        print("Success rate: {:.1f}%".format(success_rate))
        print("Average energy: {:.2f}".format(stats.get('final_energy', 0)))
        print("Average dopamine: {:.2f}".format(stats.get('avg_dopamine', 0)))
        
    except Exception as e:
        print("ERROR: {}".format(e))


def option_3_juvenile_mode():
    """OPTION 3: Juvenile mode (intense learning)"""
    print("\n" + "-"*70)
    print("JUVENILE MODE (INTENSIVE LEARNING)")
    print("-"*70)
    print("""
In juvenile mode:
  - Dopamine = 0.85 (maximum!) - strong reinforcement
  - STDP learning is highly active
  - Brain learns to catch flies quickly
  - This is a critical developmental period
  
Parameters:
  - Flies: 15
  - Steps: 3000 (intensive learning)
  - Visualization: Yes
    """)

    input("Press ENTER to start...")
    
    if not IMPORT_OK:
        print("ERROR: Could not import frog_lib\n{}".format(IMPORT_ERROR))
        return
    
    try:
        sim = BioFlyCatchingSimulation(
            width=800,
            height=600,
            num_flies=15,
            headless=False
        )
        sim.juvenile_mode = True
        
        print("\nStarting juvenile mode...\n")
        sim.run_simulation(3000)
        
        print("\nJuvenile phase completed!")
        print("Flies caught: {}".format(sim.stats.get('caught_flies', 0)))
        success_rate = sim.stats.get('catch_success_rate', 0) * 100
        print("Final success rate: {:.1f}%".format(success_rate))
        
    except Exception as e:
        print("ERROR: {}".format(e))


def option_4_adult_mode():
    """OPTION 4: Adult frog mode"""
    print("\n" + "-"*70)
    print("ADULT FROG MODE (STABILIZED BEHAVIOR)")
    print("-"*70)
    print("""
In adult mode:
  - Dopamine = 0.3 (normal) - weak reinforcement
  - STDP learning slows down (consolidation)
  - Behavior stabilizes
  - Synapses are protected from overlearning
  
Parameters:
  - Flies: 15
  - Steps: 2000
  - Visualization: Yes
    """)

    input("Press ENTER to start...")
    
    if not IMPORT_OK:
        print("ERROR: Could not import frog_lib\n{}".format(IMPORT_ERROR))
        return
    
    try:
        sim = BioFlyCatchingSimulation(
            width=800,
            height=600,
            num_flies=15,
            headless=False
        )
        sim.juvenile_mode = False
        
        print("\nStarting adult mode...\n")
        sim.run_simulation(2000)
        
        print("\nSimulation completed!")
        print("Flies caught: {}".format(sim.stats.get('caught_flies', 0)))
        success_rate = sim.stats.get('catch_success_rate', 0) * 100
        print("Success rate: {:.1f}%".format(success_rate))
        
    except Exception as e:
        print("ERROR: {}".format(e))


def option_5_components_info():
    """OPTION 5: Component information"""
    print("\n" + "-"*70)
    print("BRAIN COMPONENTS INFORMATION")
    print("-"*70)
    print("""
BioFrog v2.0 consists of 11 biological components:

CORE:
  - LIFNeuron: Leaky Integrate-and-Fire neurons
  - BiologicalSynapse: Synapses with STDP learning
  - Astrocyte: Glial cells for neuromodulation
  - NeurotransmitterDiffusion: Neural chemistry simulation

ARCHITECTURE:
  - RetinalProcessing: Visual system (ON/OFF filtering)
  - Tectum: Motion detection (4 directional selectivity)
  - MotorHierarchy: Hierarchical movement control

METABOLISM:
  - SystemicMetabolism: Energy and attention
  - Circadian: Daily rhythm regulation

PLASTICITY:
  - FunctionalPlasticity: Homeostasis and compensation
  - StructuralPlasticity: Synapse creation/deletion

All components work together to create a realistic brain!
    """)

    input("Press ENTER to return to menu...")


def option_6_diagnostics():
    """OPTION 6: Project diagnostics"""
    print("\n" + "-"*70)
    print("DIAGNOSTICS")
    print("-"*70)
    
    if not IMPORT_OK:
        print("ERROR: Could not import frog_lib")
        print("Details: {}".format(IMPORT_ERROR))
        input("Press ENTER to continue...")
        return
    
    try:
        from frog_lib import (
            BioFrogBrain, LIFNeuron, BiologicalSynapse,
            RetinalProcessing, Tectum, MotorHierarchy,
            SystemicMetabolism, FunctionalPlasticityManager,
            StructuralPlasticityManager
        )
        
        print("\nALL COMPONENTS SUCCESSFULLY LOADED:\n")
        
        components = [
            ("BioFrogBrain", "Master controller"),
            ("LIFNeuron", "Neuron model"),
            ("BiologicalSynapse", "Synaptic plasticity"),
            ("RetinalProcessing", "Visual system"),
            ("Tectum", "Motion detection"),
            ("MotorHierarchy", "Movement control"),
            ("SystemicMetabolism", "Energy management"),
            ("FunctionalPlasticityManager", "Homeostasis"),
            ("StructuralPlasticityManager", "Synapse dynamics"),
        ]
        
        for comp_name, comp_desc in components:
            print("  OK  {:<30} - {}".format(comp_name, comp_desc))
        
        print("\nDIAGNOSTICS SUCCESSFUL!")
        print("   All systems are ready to use")
        
    except Exception as e:
        print("ERROR: {}".format(e))
    
    input("\nPress ENTER to return to menu...")


def option_7_docs():
    """OPTION 7: Documentation guide"""
    print("\n" + "-"*70)
    print("DOCUMENTATION")
    print("-"*70)
    print("""
Main project documents:

DEBUG_REPORT.md
   - Full diagnostic report
   - Error explanations and solutions
   - Correct ways to run the project

PROJECT_STRUCTURE.md
   - Project structure overview
   - Module descriptions
   - Neural network architecture

QUICK_START.py
   - Code examples for beginners
   - How to use frog_lib in your code

RESTRUCTURING_COMPLETE.md
   - History of restructuring
   - What was changed and why

README_COMPLETE.md
   - General BioFrog information
   - Biological foundations
   - Testing results

TO RUN THESE SCRIPTS:
   - python run_simulation.py     (full simulation)
   - python run_biofrog_examples.py (4 examples)
   - python test_biofrog.py       (quick test)
    """)
    
    input("Press ENTER to return to menu...")


def main_menu():
    """Main menu loop"""
    if not IMPORT_OK:
        print("\nCRITICAL ERROR")
        print("="*70)
        print("Could not import frog_lib module!")
        print("Details: {}".format(IMPORT_ERROR))
        print("\nMake sure:")
        print("  1. You're running from the project root directory")
        print("  2. All files are in the frog_lib/ folder")
        print("  3. All dependencies are installed (pygame, numpy, etc.)")
        print("="*70)
        sys.exit(1)
    
    while True:
        print_header()
        
        print("""
AVAILABLE OPTIONS:
  1  Full simulation with visualization (800x600, 5000 steps)
  2  Quick test without visualization (500 steps, ~5 sec)
  3  Juvenile mode - intensive learning (dopamine 0.85)
  4  Adult mode - stabilized behavior (dopamine 0.3)
  5  Brain components information
  6  Project diagnostics (check all systems)
  7  Documentation guide
  0  Exit
        """)
        
        choice = input("Select option (0-7): ").strip()
        
        if choice == "1":
            option_1_full_simulation()
        elif choice == "2":
            option_2_quick_test()
        elif choice == "3":
            option_3_juvenile_mode()
        elif choice == "4":
            option_4_adult_mode()
        elif choice == "5":
            option_5_components_info()
        elif choice == "6":
            option_6_diagnostics()
        elif choice == "7":
            option_7_docs()
        elif choice == "0":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid option. Try again.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print("\nCRITICAL ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
