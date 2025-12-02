"""General-purpose script to test experiment configurations in isolation.

This script automatically discovers and runs experiment configurations from the
config/experiments directory. It can be used to test any component or combination
of components by simply creating a new config file.

Usage:
    # List all available configs
    python test_model_configuration.py --list
    
    # Run a specific config (by filename without .py)
    python test_model_configuration.py baseline
    python test_model_configuration.py test_adamw
    
    # Run multiple configs
    python test_model_configuration.py baseline test_adamw test_white_noise_1.0
    
    # Run all configs
    python test_model_configuration.py --all
"""
import sys
import os
import importlib.util
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add project root and src directory to path (for Colab compatibility)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'experiments'

# Add to path in order: src (for package imports), then config (for config imports)
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from neural_decoder.neural_decoder_trainer import trainModel


# Environment path mappings
ENV_PATHS = {
    'kaggle': {
        'base': '/kaggle/working/neural_seq_decoder',
    },
    'google_cloud': {
        'base': '/content/drive/MyDrive/UCLA_classes/ece243a/neural_seq_decoder',
    },
    'gcp_instance': {
        'base': '/home/latka/github/neural_seq_decoder',
    },
}


def adjust_paths_for_environment(args: Dict, env: str) -> Dict:
    """Adjust outputDir and datasetPath in config based on environment.
    
    Args:
        args: Configuration dictionary
        env: Environment name ('kaggle' or 'google_cloud')
        
    Returns:
        Modified args dictionary with adjusted paths
    """
    if env not in ENV_PATHS:
        raise ValueError(f"Unknown environment: {env}. Choose from: {list(ENV_PATHS.keys())}")
    
    target_base = ENV_PATHS[env]['base']
    
    # Create a copy to avoid modifying the original
    adjusted_args = args.copy()
    
    def replace_base_path(path: str) -> str:
        """Replace any known base path with the target environment's base path."""
        # Try each known base path and replace if found
        for env_name, env_config in ENV_PATHS.items():
            old_base = env_config['base']
            if path.startswith(old_base):
                # Extract relative path and prepend target base
                relative_path = path[len(old_base):].lstrip('/')
                return f"{target_base}/{relative_path}" if relative_path else target_base
        # If no known base found, return as-is (might be a custom path)
        return path
    
    # Adjust outputDir if it exists
    if 'outputDir' in adjusted_args:
        adjusted_args['outputDir'] = replace_base_path(adjusted_args['outputDir'])
    
    # Adjust datasetPath if it exists
    if 'datasetPath' in adjusted_args:
        adjusted_args['datasetPath'] = replace_base_path(adjusted_args['datasetPath'])
    
    return adjusted_args


def discover_configs() -> Dict[str, Path]:
    """Discover all available configuration files.
    
    Returns:
        Dictionary mapping config names (without .py) to file paths
    """
    configs = {}
    
    if not CONFIG_DIR.exists():
        return configs
    
    for config_file in CONFIG_DIR.glob('*.py'):
        # Skip __init__.py and other special files
        if config_file.name.startswith('_'):
            continue
        
        # Get config name without extension
        config_name = config_file.stem
        configs[config_name] = config_file
    
    return configs


def load_config(config_name: str) -> Dict:
    """Load configuration from a config file.
    
    Args:
        config_name: Name of config file (with or without .py extension)
        
    Returns:
        args dictionary from the config file
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file doesn't define 'args'
    """
    # Remove .py extension if present
    if config_name.endswith('.py'):
        config_name = config_name[:-3]
    
    configs = discover_configs()
    
    if config_name not in configs:
        available = ', '.join(sorted(configs.keys()))
        raise FileNotFoundError(
            f"Config '{config_name}' not found.\n"
            f"Available configs: {available}\n"
            f"Use --list to see all available configs."
        )
    
    config_file = configs[config_name]
    
    # Load the config module
    spec = importlib.util.spec_from_file_location(config_name, config_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config file: {config_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check if args is defined
    if not hasattr(module, 'args'):
        raise ValueError(
            f"Config file '{config_file}' must define an 'args' dictionary."
        )
    
    return module.args


def get_config_info(config_name: str) -> Dict[str, Optional[str]]:
    """Extract metadata from a config file.
    
    Args:
        config_name: Name of config file
        
    Returns:
        Dictionary with config metadata (description, model_name, etc.)
    """
    try:
        config_file = discover_configs()[config_name]
        
        # Read file to extract docstring and modelName
        with open(config_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        info = {
            'name': config_name,
            'file': str(config_file),
            'description': None,
            'model_name': None,
        }
        
        # Extract docstring (first string after module docstring)
        if '"""' in content:
            docstring_start = content.find('"""') + 3
            docstring_end = content.find('"""', docstring_start)
            if docstring_end > docstring_start:
                info['description'] = content[docstring_start:docstring_end].strip()
        
        # Extract modelName if present
        for line in lines:
            if 'modelName' in line and '=' in line:
                try:
                    # Extract value from modelName = 'value'
                    # Remove comments first, then extract value
                    line_without_comment = line.split('#')[0]  # Remove comment
                    value = line_without_comment.split('=')[1].strip().strip("'\"")
                    info['model_name'] = value
                    break
                except:
                    pass
        
        return info
    except Exception:
        return {'name': config_name, 'description': None, 'model_name': None}


def list_configs():
    """List all available configuration files with their descriptions."""
    configs = discover_configs()
    
    if not configs:
        print("No configuration files found in config/experiments/")
        return
    
    print(f"\n{'='*70}")
    print(f"Available Experiment Configurations ({len(configs)} found)")
    print(f"{'='*70}\n")
    
    for config_name in sorted(configs.keys()):
        info = get_config_info(config_name)
        print(f"  {config_name}")
        if info.get('model_name'):
            print(f"    Model Name: {info['model_name']}")
        if info.get('description'):
            # Truncate long descriptions
            desc = info['description']
            if len(desc) > 60:
                desc = desc[:57] + '...'
            print(f"    Description: {desc}")
        print()


def test_config(config_name: str, verbose: bool = True, env: Optional[str] = None) -> bool:
    """Test a specific configuration.
    
    Args:
        config_name: Name of config to test
        verbose: Whether to print progress messages
        env: Environment name ('kaggle' or 'google_cloud') to adjust paths
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing Configuration: {config_name}")
            if env:
                print(f"Environment: {env}")
            print(f"{'='*70}\n")
        
        args = load_config(config_name)
        
        # Adjust paths for environment if specified
        if env:
            args = adjust_paths_for_environment(args, env)
            if verbose:
                print(f"Adjusted paths for {env}:")
                if 'outputDir' in args:
                    print(f"  outputDir: {args['outputDir']}")
                if 'datasetPath' in args:
                    print(f"  datasetPath: {args['datasetPath']}")
                print()
        
        # Run training
        trainModel(args)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✓ Completed: {config_name}")
            print(f"{'='*70}\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return False
    except ValueError as e:
        print(f"\n✗ Error loading config '{config_name}': {e}\n", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n✗ Error running config '{config_name}': {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test experiment configurations in isolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available configs
  python test_model_configuration.py --list
  
  # Run a specific config
  python test_model_configuration.py baseline
  python test_model_configuration.py test_adamw
  
  # Run with environment-specific paths
  python test_model_configuration.py baseline --env kaggle
  python test_model_configuration.py baseline --env google_cloud
  python test_model_configuration.py baseline --env gcp_instance
  
  # Run multiple configs
  python test_model_configuration.py baseline test_adamw
  
  # Run all configs
  python test_model_configuration.py --all
        """
    )
    
    parser.add_argument(
        'configs',
        nargs='*',
        help='Configuration names to test (omit to see help)'
    )
    
    parser.add_argument(
        '--list',
        '-l',
        action='store_true',
        help='List all available configurations'
    )
    
    parser.add_argument(
        '--all',
        '-a',
        action='store_true',
        help='Run all available configurations'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    parser.add_argument(
        '--env',
        '-e',
        type=str,
        choices=['kaggle', 'google_cloud', 'gcp_instance'],
        default=None,
        help='Environment to use for path adjustment (kaggle, google_cloud, or gcp_instance)'
    )
    
    args = parser.parse_args()
    
    # List configs if requested
    if args.list:
        list_configs()
        return
    
    # Get configs to run
    if args.all:
        configs_to_run = sorted(discover_configs().keys())
        if not configs_to_run:
            print("No configuration files found in config/experiments/")
            return
    elif args.configs:
        configs_to_run = args.configs
    else:
        parser.print_help()
        print("\nUse --list to see available configurations.")
        return
    
    # Run configs
    if not args.quiet and len(configs_to_run) > 1:
        print(f"\nRunning {len(configs_to_run)} configuration(s)...\n")
    
    results = {}
    for config_name in configs_to_run:
        success = test_config(config_name, verbose=not args.quiet, env=args.env)
        results[config_name] = success
    
    # Summary
    if len(configs_to_run) > 1:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        for config_name in configs_to_run:
            status = "✓ Success" if results[config_name] else "✗ Failed"
            print(f"  {config_name}: {status}")
        print()


if __name__ == "__main__":
    main()
