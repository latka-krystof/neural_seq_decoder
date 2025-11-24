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

# Add config directory to path
CONFIG_DIR = Path(__file__).parent.parent / 'config' / 'experiments'
sys.path.insert(0, str(CONFIG_DIR))

from neural_decoder.neural_decoder_trainer import trainModel


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


def test_config(config_name: str, verbose: bool = True) -> bool:
    """Test a specific configuration.
    
    Args:
        config_name: Name of config to test
        verbose: Whether to print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing Configuration: {config_name}")
            print(f"{'='*70}\n")
        
        args = load_config(config_name)
        
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
        success = test_config(config_name, verbose=not args.quiet)
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
