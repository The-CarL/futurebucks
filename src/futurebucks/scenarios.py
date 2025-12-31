"""Scenario loading, saving, and management."""

import shutil
from pathlib import Path

import yaml

from futurebucks.models import Scenario


def get_scenarios_dir() -> Path:
    """Get the scenarios directory path."""
    return Path(__file__).parent.parent.parent / "scenarios"


def get_sample_scenarios_dir() -> Path:
    """Get the sample scenarios directory path."""
    return Path(__file__).parent.parent.parent / "samples"


def list_scenarios() -> list[str]:
    """List all available scenario files.

    Returns:
        List of scenario file names (without path)
    """
    scenarios_dir = get_scenarios_dir()
    if not scenarios_dir.exists():
        return []

    return sorted(
        f.name
        for f in scenarios_dir.glob("*.yaml")
        if f.is_file()
    ) + sorted(
        f.name
        for f in scenarios_dir.glob("*.yml")
        if f.is_file()
    )


def list_sample_scenarios() -> list[str]:
    """List all available sample scenario files.

    Returns:
        List of sample scenario file names (without path)
    """
    sample_dir = get_sample_scenarios_dir()
    if not sample_dir.exists():
        return []

    return sorted(
        f.name
        for f in sample_dir.glob("*.yaml")
        if f.is_file()
    ) + sorted(
        f.name
        for f in sample_dir.glob("*.yml")
        if f.is_file()
    )


def load_scenario(filename: str, from_samples: bool = False) -> Scenario:
    """Load a scenario from a YAML file.

    Args:
        filename: Name of the scenario file
        from_samples: If True, load from samples directory

    Returns:
        Loaded Scenario object

    Raises:
        FileNotFoundError: If scenario file doesn't exist
        ValueError: If scenario file is invalid
    """
    if from_samples:
        filepath = get_sample_scenarios_dir() / filename
    else:
        filepath = get_scenarios_dir() / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Scenario file not found: {filepath}")

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty scenario file: {filepath}")

    return Scenario.model_validate(data)


def save_scenario(scenario: Scenario, filename: str) -> Path:
    """Save a scenario to a YAML file.

    Args:
        scenario: Scenario to save
        filename: Name of the file to save to

    Returns:
        Path to saved file
    """
    scenarios_dir = get_scenarios_dir()
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    filepath = scenarios_dir / filename
    if not filepath.suffix:
        filepath = filepath.with_suffix(".yaml")

    data = scenario.model_dump(exclude_none=True)

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return filepath


def copy_samples_to_scenarios() -> list[str]:
    """Copy all sample scenarios to the scenarios directory.

    Returns:
        List of copied file names
    """
    sample_dir = get_sample_scenarios_dir()
    scenarios_dir = get_scenarios_dir()

    if not sample_dir.exists():
        return []

    scenarios_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for sample_file in sample_dir.glob("*.yaml"):
        dest = scenarios_dir / sample_file.name
        if not dest.exists():
            shutil.copy(sample_file, dest)
            copied.append(sample_file.name)

    for sample_file in sample_dir.glob("*.yml"):
        dest = scenarios_dir / sample_file.name
        if not dest.exists():
            shutil.copy(sample_file, dest)
            copied.append(sample_file.name)

    return copied


def scenario_to_yaml(scenario: Scenario) -> str:
    """Convert a scenario to YAML string.

    Args:
        scenario: Scenario to convert

    Returns:
        YAML string representation
    """
    data = scenario.model_dump(exclude_none=True)
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def yaml_to_scenario(yaml_str: str) -> Scenario:
    """Parse a YAML string into a Scenario.

    Args:
        yaml_str: YAML string to parse

    Returns:
        Parsed Scenario object

    Raises:
        ValueError: If YAML is invalid
    """
    try:
        data = yaml.safe_load(yaml_str)
        if data is None:
            raise ValueError("Empty YAML content")
        return Scenario.model_validate(data)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e


def delete_scenario(filename: str) -> bool:
    """Delete a scenario file.

    Args:
        filename: Name of scenario file to delete

    Returns:
        True if deleted, False if file didn't exist
    """
    filepath = get_scenarios_dir() / filename
    if filepath.exists():
        filepath.unlink()
        return True
    return False
