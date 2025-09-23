"""
Launch Script for Property Similarity Search
Provides easy startup for different components of the system
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

def check_requirements():
    """Check if required files and dependencies exist"""
    print("🔍 Checking system requirements...")

    # Check if embeddings exist
    if not os.path.exists('embeddings'):
        print("⚠️  Embeddings folder not found!")
        print("   Run: python build_embeddings.py --sample 100000")
        return False

    # Check if dataset exists
    if not os.path.exists('realtor_cleaned_final.csv'):
        print("⚠️  Dataset not found!")
        print("   Please ensure realtor_cleaned_final.csv is in the project directory")
        return False

    # Check if key files exist
    required_files = [
        'property_search_engine.py',
        'streamlit_app.py',
        'api.py'
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file missing: {file}")
            return False

    print("✅ All requirements met!")
    return True

def launch_streamlit():
    """Launch the Streamlit web application"""
    print("🚀 Starting Streamlit web application...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")

def launch_api():
    """Launch the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop\n")

    try:
        subprocess.run([
            sys.executable, "api.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API: {e}")

def run_demo():
    """Run the quick demo"""
    print("🎯 Running quick demo...")

    try:
        subprocess.run([
            sys.executable, "quick_demo.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")

def run_tests():
    """Run KPI compliance tests"""
    print("🧪 Running KPI compliance tests...")

    try:
        subprocess.run([
            sys.executable, "test_kpi_compliance.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tests: {e}")

def build_embeddings(sample_size=None):
    """Build embeddings with optional sample size"""
    print("🏗️  Building embeddings...")

    cmd = [sys.executable, "build_embeddings.py"]
    if sample_size:
        cmd.extend(["--sample", str(sample_size)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building embeddings: {e}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")

def show_status():
    """Show system status"""
    print("📊 Property Similarity Search - System Status")
    print("=" * 50)

    # Check Python version
    print(f"🐍 Python: {sys.version.split()[0]}")

    # Check files
    files_status = {
        "Dataset (CSV)": os.path.exists('realtor_cleaned_final.csv'),
        "Search Engine": os.path.exists('property_search_engine.py'),
        "Streamlit App": os.path.exists('streamlit_app.py'),
        "API Server": os.path.exists('api.py'),
        "Embeddings": os.path.exists('embeddings'),
        "Requirements": os.path.exists('requirements.txt')
    }

    for item, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"{status} {item}")

    # Check embeddings details
    if os.path.exists('embeddings'):
        embedding_files = list(Path('embeddings').glob('*'))
        print(f"   📁 Embedding files: {len(embedding_files)}")

    print("\n🚀 Available Commands:")
    print("   python launch_app.py web      - Start web interface")
    print("   python launch_app.py api      - Start API server")
    print("   python launch_app.py demo     - Run quick demo")
    print("   python launch_app.py test     - Run tests")
    print("   python launch_app.py build    - Build embeddings")
    print("   python launch_app.py install  - Install dependencies")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Property Similarity Search Launcher")
    parser.add_argument(
        'command',
        choices=['web', 'api', 'demo', 'test', 'build', 'install', 'status'],
        help='Command to run'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Sample size for building embeddings (e.g., 100000)'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip requirements check'
    )

    args = parser.parse_args()

    print("🏠 Property Similarity Search Launcher")
    print("=" * 40)

    # Handle status command first (doesn't need checks)
    if args.command == 'status':
        show_status()
        return

    # Handle install command (doesn't need full checks)
    if args.command == 'install':
        install_dependencies()
        return

    # Handle build command (doesn't need embeddings check)
    if args.command == 'build':
        build_embeddings(args.sample)
        return

    # For other commands, check requirements
    if not args.skip_check and not check_requirements():
        print("\n💡 Try running:")
        print("   python launch_app.py install  # Install dependencies")
        print("   python launch_app.py build --sample 100000  # Build embeddings")
        return

    # Execute the requested command
    if args.command == 'web':
        launch_streamlit()
    elif args.command == 'api':
        launch_api()
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'test':
        run_tests()

if __name__ == "__main__":
    main()