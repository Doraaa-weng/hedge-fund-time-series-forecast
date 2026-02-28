#!/usr/bin/env python3
"""
Interactive package installation script
Tests connectivity and installs required packages
"""
import subprocess
import sys
import socket

def test_connectivity():
    """Test if we can reach the internet"""
    print("Testing network connectivity...")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("✓ Network connection OK")
        return True
    except OSError:
        print("✗ Network connection failed")
        return False

def install_with_conda():
    """Try installing with conda"""
    print("\nAttempting installation with conda...")
    packages = ['pandas', 'pyarrow', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    cmd = ['conda', 'install'] + packages + ['-y']
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Conda installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conda installation failed: {e.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("✗ Conda not found")
        return False

def install_with_pip():
    """Try installing with pip"""
    print("\nAttempting installation with pip...")
    packages = ['pandas', 'pyarrow', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Pip installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Pip installation failed: {e.stderr[:200]}")
        return False

def verify_installation():
    """Verify that packages are installed"""
    print("\nVerifying installation...")
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'pyarrow': 'pyarrow'
    }
    
    all_ok = True
    for module, package_name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("Package Installation Script")
    print("=" * 60)
    
    # Test connectivity
    if not test_connectivity():
        print("\n⚠ WARNING: Network connectivity test failed!")
        print("Please check your internet connection and try again.")
        print("\nYou can also install packages manually:")
        print("  conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y")
        print("  OR")
        print("  pip install pandas pyarrow numpy scikit-learn matplotlib seaborn")
        return False
    
    # Try conda first
    if install_with_conda():
        if verify_installation():
            print("\n" + "=" * 60)
            print("✓ All packages installed successfully!")
            print("=" * 60)
            return True
    
    # Fallback to pip
    if install_with_pip():
        if verify_installation():
            print("\n" + "=" * 60)
            print("✓ All packages installed successfully!")
            print("=" * 60)
            return True
    
    print("\n" + "=" * 60)
    print("✗ Installation failed. Please install packages manually.")
    print("See INSTALLATION_GUIDE.md for detailed instructions.")
    print("=" * 60)
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
