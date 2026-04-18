#!/bin/bash

# run.sh - Simple analysis runner
# Usage: bash run.sh

# Check if virtual environment exists and use it if available
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Using virtual environment"
elif [ -z "$CONDA_PREFIX" ]; then
    echo "Note: No virtual environment or Anaconda detected"
    echo "      Using system Python"
fi

echo "============================================================"
echo "      Running Maintainability Analysis Pipeline"
echo "============================================================"
echo ""

# Run the full pipeline
echo "Starting analysis... (this may take 15-30 minutes)"
echo ""
python main.py --repo test_repos/flask --run-all

echo ""
echo "============================================================"
echo "                 Analysis Complete!"
echo "============================================================"
echo ""
echo "Results saved to: data/results/"
echo ""
echo "View the interactive report:"
echo "    data/results/maintainability_report.html"
echo ""
echo "JSON results available:"
echo "    • ga_results.json"
echo "    • baseline_results.json"
echo "    • ablation_results.json"
echo "    • stats_results.json"
echo ""
echo "============================================================"
echo "                    Next Steps"
echo "============================================================"
echo ""
echo "To launch the interactive dashboard, run:"
echo ""
echo "         bash start.sh"
echo ""

