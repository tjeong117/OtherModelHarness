"""Export an offline tiny-target task for Harbor 0.21's Docker environment."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


# This allowlist defines the contents of the ordinary agent's build context.
# New CLI commands, graders and oracle recipes belong outside it by default.
_PUBLIC_MODULES = ("__init__", "targets", "probes", "causal", "dictionaries", "tokenization", "weights", "tools",
                   "experiments", "preview", "session")


def make_task(output_dir: str | Path) -> Path:
    """Materialize a self-contained task into a new or empty directory."""
    dest = Path(output_dir).absolute()
    if dest.is_symlink() or (dest.exists() and (not dest.is_dir() or any(dest.iterdir()))):
        raise FileExistsError(f"Harbor task destination must be new or empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    template = package / "harbor_template"
    with TemporaryDirectory(prefix=".iwi-task-", dir=dest.parent) as tmp:
        task = Path(tmp) / "task"
        task.mkdir()
        for name in ("task.toml", "instruction.md"):
            shutil.copyfile(template / name, task / name)
        for area in ("environment", "solution", "tests"):
            shutil.copytree(template / area, task / area, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            source_dir = task / area / "iwi_harness"
            source_dir.mkdir()
            if area == "environment":
                source_files = [package / f"{name}.py" for name in _PUBLIC_MODULES]
            else:
                source_files = sorted(package.glob("*.py"))
                bin_dir = task / area / "bin"
                bin_dir.mkdir()
                command = bin_dir / "iwi-harness"
                shutil.copyfile(template / "cli-entrypoint", command)
                command.chmod(0o755)
            for source in source_files:
                shutil.copyfile(source, source_dir / source.name)

        # Harbor builds the verifier from tests/, a separate Docker context.
        docker_base = (template / "base.Dockerfile").read_text()
        for area in ("environment", "tests"):
            dockerfile = task / area / "Dockerfile"
            dockerfile.write_text(docker_base + "\n" + dockerfile.read_text())
        for command in ("environment/edit-target", "solution/solve.sh", "tests/test.sh"):
            (task / command).chmod(0o755)
        task.rename(dest)
    return dest
