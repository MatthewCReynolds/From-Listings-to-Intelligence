"""Colored console messages used in notebooks."""
BLUE, GREEN, YELLOW, RED, BOLD, RESET = "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"


def step(msg):
    """Print a prominent step heading."""
    print(f"\n{BOLD}{BLUE}▶ {msg}{RESET}")


def done(msg):
    """Print a completion message."""
    print(f"{GREEN}  ✓ {msg}{RESET}")


def note(msg):
    """Print an informational note."""
    print(f"{YELLOW}  • {msg}{RESET}")
