set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/experiments/train.py" --domain science --size 1.7b --method RuACT-Critic "$@"
