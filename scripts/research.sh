#!/usr/bin/env bash
# scripts/research.sh — Enhanced CLI helper for typical research workflows.
# This script runs inside the Docker container to simplify common tasks.

set -e

# Default repos
REPOS=("flask" "requests" "django")
REPO_BASE="test_repos"

_banner() {
    echo -e "\n\033[1;34m============================================================\033[0m"
    echo -e "\033[1;32m  $1\033[0m"
    echo -e "\033[1;34m============================================================\033[0m\n"
}

_prep_repos() {
    mkdir -p $REPO_BASE
    for repo in "${REPOS[@]}"; do
        if [ ! -d "$REPO_BASE/$repo" ]; then
            echo "▶ Cloning $repo..."
            git clone --depth 1 "https://github.com/$( [[ $repo == "flask" ]] && echo "pallets/flask" || ([[ $repo == "requests" ]] && echo "psf/requests" || echo "django/django") ).git" "$REPO_BASE/$repo"
        else
            echo "✓ $repo already exists."
        fi
    done
}

case "$1" in
    --replicate-paper)
        _banner "REPLICATING BASE PAPER CONDITIONS (Accuracy Priority)"
        python main.py --alpha 2.0 --beta 0.1 --generations 15 --run-all
        ;;
    --parsimony-experiment)
        _banner "EXTREME PARSIMONY EXPERIMENT (Efficiency Priority)"
        python main.py --alpha 0.5 --beta 2.0 --generations 15 --run-all
        ;;
    --multi-repo)
        _banner "MULTI-REPOSITORY GENERALISATION STUDY"
        _prep_repos
        python main.py --multi-repo --repos "${REPOS[@]}" --run-baselines --run-report
        ;;
    --tune-and-run)
         _banner "TUNING + FULL PIPELINE RUN"
         python main.py --repo test_repos/flask --run-tuning --run-all
         ;;
    --force-all)
        _banner "FORCE COMPLETE RE-RUN"
        python main.py --run-all --force-all
        ;;
    *)
        echo "Usage: ./scripts/research.sh [command]"
        echo "Commands:"
        echo "  --replicate-paper      Run GA with alpha=2.0, beta=0.1"
        echo "  --parsimony-experiment Run GA with alpha=0.5, beta=2.0"
        echo "  --multi-repo           Clone and study Flask, Requests, Django"
        echo "  --tune-and-run         Tune ANN then run full pipeline on Flask"
        echo "  --force-all            Force re-run all stages"
        echo ""
        echo "Alternatively, run 'python main.py --help' for fine-grained control."
        exit 1
        ;;
esac
