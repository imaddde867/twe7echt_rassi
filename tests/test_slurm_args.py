"""The argument block of slurm/train_es.sbatch, run on its own: positional GENERATIONS, an optional resume
directory, then `--seed N` and `--reward KEY=VALUE ...`. The block sits between two marker comments in the script."""
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "slurm" / "train_es.sbatch"


def _block() -> str:
    m = re.search(r"# --- args ---\n(.*?)# --- end args ---", SCRIPT.read_text(), re.S)
    assert m, "train_es.sbatch has no '# --- args ---' ... '# --- end args ---' block"
    return m.group(1)


def parse(*args: str):
    """Run the block with these arguments; return (exit code, GENERATIONS, RESUME, SEED_ARGS, REWARD_ARGS, stderr)."""
    show = ('printf "%s\\n" "$GENERATIONS" "$RESUME"; '
            'echo "${SEED_ARGS[*]-}"; echo "${REWARD_ARGS[*]-}"')
    r = subprocess.run(["bash", "-c", f"set -euo pipefail\n{_block()}\n{show}", "bash", *args],
                       capture_output=True, text=True)
    out = r.stdout.split("\n")
    if r.returncode:
        return r.returncode, None, None, None, None, r.stderr
    return r.returncode, out[0], out[1], out[2], out[3], r.stderr


def test_no_arguments_keeps_the_old_defaults():
    assert parse()[:5] == (0, "300", "", "", "")


def test_generations_alone_and_with_a_resume_directory_still_work():
    assert parse("20")[:5] == (0, "20", "", "", "")
    assert parse("300", "runs/rl_2004461")[:5] == (0, "300", "runs/rl_2004461", "", "")


def test_seed_goes_to_the_trainer_and_does_not_look_like_a_resume_directory():
    assert parse("20", "--seed", "3")[:5] == (0, "20", "", "--seed 3", "")
    assert parse("300", "runs/rl_x", "--seed", "3")[:5] == (0, "300", "runs/rl_x", "--seed 3", "")


def test_reward_takes_every_key_value_up_to_the_next_option():
    got = parse("20", "--reward", "w_wealth=0.25", "cash_scale=5000", "--seed", "3")
    assert got[:5] == (0, "20", "", "--seed 3", "--reward w_wealth=0.25 cash_scale=5000")


@pytest.mark.parametrize("bad", [["20", "--bogus"], ["20", "--seed"], ["20", "stray", "extra"]])
def test_unknown_or_incomplete_arguments_stop_the_job(bad):
    code, *_, err = parse(*bad)
    assert code == 2 and "train_es.sbatch" in err
