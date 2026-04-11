#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launcher for BioFrog example scenarios."""

from frog_lib.examples import (
    example_1_basic_simulation,
    example_2_adult_mode,
    example_3_headless,
    example_4_compare_modes,
)


def main():
    print("\nBioFrog examples")
    print("1. Juvenile visual run")
    print("2. Adult visual run")
    print("3. Headless quick run")
    print("4. Juvenile vs adult comparison")
    print("0. Exit")

    while True:
        choice = input("Select example (0-4): ").strip()
        if choice == "1":
            example_1_basic_simulation()
        elif choice == "2":
            example_2_adult_mode()
        elif choice == "3":
            example_3_headless()
        elif choice == "4":
            example_4_compare_modes()
        elif choice == "0":
            return
        else:
            print("Invalid choice.")
            continue

        print("\n" + "-" * 68)
        if input("Run another example? (y/n): ").strip().lower() != "y":
            return


if __name__ == "__main__":
    main()
