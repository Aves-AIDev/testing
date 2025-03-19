import subprocess
import os
import sys
import platform
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

def print_colored(text, color=Fore.GREEN, style=Style.BRIGHT):
    """Print colored text"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def run_command(command, description):
    """Run a command and print the output"""
    print_colored(f"\n>>> {description}...", Fore.CYAN)
    print_colored(f"    Running: {command}", Fore.YELLOW, Style.DIM)
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print_colored(f"    ✓ Success!", Fore.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"    ✗ Error: {e.stderr}", Fore.RED)
        return False

def main():
    """Install dependencies and configure the application"""
    print_colored("\n=== PDF RAG Chatbot Setup ===", Fore.BLUE, Style.BRIGHT)
    
    # Check if virtual environment exists, create if not
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print_colored("\nCreating virtual environment...", Fore.CYAN)
        venv_command = "python -m venv .venv"
        if not run_command(venv_command, "Creating virtual environment"):
            print_colored("Failed to create virtual environment. Exiting.", Fore.RED)
            return
    
    # Activate virtual environment
    if platform.system() == "Windows":
        activate_cmd = f"{venv_dir}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_dir}/bin/activate"
    
    # Install requirements
    install_req = f"{activate_cmd} && pip install -r requirements.txt"
    if not run_command(install_req, "Installing requirements"):
        print_colored("Failed to install requirements. Trying to install one by one...", Fore.YELLOW)
        
        # Try to install individual packages
        with open("requirements.txt", "r") as f:
            packages = f.read().splitlines()
        
        for package in packages:
            if package.strip() and not package.startswith("#"):
                install_pkg = f"{activate_cmd} && pip install {package}"
                run_command(install_pkg, f"Installing {package}")
    
    # Install additional dependencies for HuggingFace embeddings
    hf_install = f"{activate_cmd} && pip install -U sentence-transformers langchain-huggingface"
    run_command(hf_install, "Installing HuggingFace dependencies")
    
    # Check if .env file exists, create if not
    if not os.path.exists(".env"):
        print_colored("\nCreating .env file...", Fore.CYAN)
        with open(".env", "w") as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
            f.write("HF_TOKEN=your_huggingface_token_here\n")
            f.write("# Uncomment and set this if HuggingFace embeddings don't work\n")
            f.write("# OPENAI_API_KEY=your_openai_api_key_here\n")
        print_colored("Created .env file. Please edit it with your API keys.", Fore.GREEN)
    
    # Create pdf directory if it doesn't exist
    pdf_dir = "pdf"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print_colored(f"Created '{pdf_dir}' directory. Please add your PDF files here.", Fore.GREEN)
    
    # Final instructions
    print_colored("\n=== Setup Complete ===", Fore.BLUE, Style.BRIGHT)
    print_colored("\nTo run the chatbot:", Fore.CYAN)
    print_colored(f"1. Edit the .env file with your API keys", Fore.WHITE)
    print_colored(f"2. Add your PDF files to the '{pdf_dir}' directory", Fore.WHITE)
    print_colored(f"3. Run: {activate_cmd} && streamlit run app.py", Fore.WHITE)
    
    # Suppress PyTorch warnings
    print_colored("\nAdding environment variables to suppress PyTorch warnings...", Fore.CYAN)
    with open(".env", "a") as f:
        f.write("\n# Suppress PyTorch warnings\n")
        f.write("PYTHONWARNINGS=ignore::UserWarning\n")
        f.write("TOKENIZERS_PARALLELISM=false\n")
    
    print_colored("\nSetup completed successfully!", Fore.GREEN, Style.BRIGHT)

if __name__ == "__main__":
    main() 