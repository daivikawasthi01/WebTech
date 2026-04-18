# Quick Start Guide (For Non-Technical Users)

## One-Step Setup

Just run this command once to install everything:

```bash
bash setup.sh
```

That's it! This will:
- Create a virtual environment
- Install all required libraries
- Download test data
- Verify everything works

## Run the Analysis

After setup, run:

```bash
bash run.sh
```

This will:
- Analyze the Flask repository
- Generate results in `data/results/`
- Create an interactive HTML report

**Expected time**: 15-30 minutes

## View Results

### Option 1: Static HTML Report (No Server)
Open this file in your browser:
```
data/results/maintainability_report.html
```

This shows:
- Feature importance
- Performance metrics
- Genetic algorithm evolution
- Comparison with baseline methods

### Option 2: Interactive Live Dashboard
After `bash run.sh` completes, launch the dashboard:

```bash
bash start.sh
```

Then open your browser to:
```
http://localhost:5173
```

Features:
- Adjust GA hyperparameters
- Re-run analysis with different settings
- Live parameter tuning
- Download results as JSON

## Troubleshooting

### Python not found?
```bash
python3 --version  # Should show 3.9 or higher
```

### Permission denied on setup.sh?
```bash
chmod +x setup.sh run.sh
bash setup.sh
```

### Want to analyze a different repository?
Edit `run.sh` and change this line:
```bash
python main.py --repo test_repos/flask --run-all
```

Replace `test_repos/flask` with your repository path.

## Need More Control?

See [README.md](README.md) for detailed CLI options and advanced workflows.

---

## Learn More

Once you're comfortable with the basics:

- **[README.md](README.md)** - Complete reference guide with all commands and options
- **[README_DEEP_DIVE.md](README_DEEP_DIVE.md)** - Technical architecture and implementation details
- **[RESEARCH_README.md](RESEARCH_README.md)** - Research methodology and academic details

---

**Roll Numbers**: 2023UIT3079, 2023UIT3062
