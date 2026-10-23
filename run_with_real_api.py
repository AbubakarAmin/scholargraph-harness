"""
Run the multi-agent research system with real API calls.
Includes rate limiting to handle the 10 requests per minute limit.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Setup environment for real API usage."""
    print("🔧 Setting up environment for real API usage...")
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key:")
        print("   export GOOGLE_API_KEY=your_api_key_here")
        print("   or")
        print("   set GOOGLE_API_KEY=your_api_key_here")
        return False
    
    # Set reasonable limits for real API usage
    os.environ["MAX_ITERATIONS"] = "2"  # Limit iterations to avoid excessive API calls
    os.environ["SUPERVISOR_THRESHOLD"] = "7.0"
    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    
    print("✅ Environment configured for real API usage")
    print("⚠️  Rate limiting enabled: 6 seconds between API calls")
    print("⚠️  Max iterations limited to 2 to manage API usage")
    return True

def main():
    """Run the system with real API calls."""
    print("🤖 Multi-Agent AI Research System - REAL API MODE")
    print("=" * 60)
    print("This will make real API calls to Gemini.")
    print("Rate limiting is enabled to respect the 10 requests/minute limit.")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Import and run main
    try:
        import main
        exit_code = main.main()
        print(f"\n🎉 System completed with exit code: {exit_code}")
        return exit_code
    except Exception as e:
        print(f"❌ System failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 