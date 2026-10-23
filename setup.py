#!/usr/bin/env python3
"""
Setup script for the multi-agent research system.
Helps users install dependencies and set up the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = [
        "output",
        "memory",
        "templates",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created {directory}/")
    
    return True

def setup_environment():
    """Set up environment file."""
    print("\nSetting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("Created .env file from template")
        print("Please edit .env and add your API keys:")
        print("   - GOOGLE_API_KEY (from https://makersuite.google.com/app/apikey)")
        print("   - OPENALEX_EMAIL (your email for rate limiting - no API key needed)")
    elif env_file.exists():
        print(".env file already exists")
    else:
        print("env_example.txt not found")
        return False
    
    return True

def run_tests():
    """Run the maintained offline pytest suite."""
    print("\nRunning offline pytest suite...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Offline pytest suite passed")
            return True
        else:
            print("Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False

def main():
    """Main setup function."""
    print("Setting up Multi-Agent Research System")
    print("=" * 50)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Running tests", run_tests)
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            success = False
            print(f"✗ {step_name} failed")
            break
    
    print("\n" + "=" * 50)
    if success:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: python main.py")
        print("3. Or test individual components:")
        print("   - python agents/topic_hunter.py")
        print("   - python agents/hypothesis_debate.py")
        print("   - python agents/planner.py")
    else:
        print("Setup failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 