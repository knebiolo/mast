# Frequently Asked Questions

## General

### What is MAST?

MAST (Movement Analysis Software for Telemetry) is a Python package for processing radio telemetry data from fish tracking studies. It identifies and removes false positive detections using a Naive Bayes classifier and processes data for movement analysis.

### Who should use MAST?

- Fisheries biologists and researchers
- Telemetry project managers
- Anyone analyzing radio telemetry data with false positive issues

### What receiver types does MAST support?

- Lotek: SRX600, SRX800, SRX1200
- Sigma Eight: Orion
- Advanced Telemetry Systems: ARES
- Vemco: VR2 (acoustic telemetry)

### Is MAST free?

Yes! MAST is open source under the MIT License. You can use it freely for academic or commercial projects.

## Installation

### What version of Python do I need?

Python 3.9 or higher. We recommend Python 3.10 or 3.11.

### Can I use MAST without coding experience?

Basic Python knowledge is helpful, but the example scripts and notebook are designed for users new to Python. Start with the Quick Start tutorial.

### Does MAST work on my operating system?

Yes! MAST works on Windows, macOS, and Linux.

## Data Requirements

### What input files do I need?

Three CSV files:
1. **tblMasterTag.csv** - Tag information (freq_code, pulse_rate, release info, etc.)
2. **tblMasterReceiver.csv** - Receiver information (rec_id, rec_type, node)
3. **tblNodes.csv** - Network nodes (node, X, Y coordinates)

See the API Reference for complete schema details.

### Do I need beacon tags?

**Strongly recommended.** Beacon tags provide known true positive detections for training the classifier. Without them, classification quality will be poor.

### How much data can MAST handle?

MAST uses HDF5 for efficient storage and Dask for parallel processing. Projects with millions of detections are feasible with adequate RAM (8+ GB recommended).

### What if my receiver type isn't supported?

You can:
1. Request support by opening a GitHub issue
2. Contribute a parser (see CONTRIBUTING.md)
3. Convert your data to match an existing format

## Classification

### How does the Naive Bayes classifier work?

The classifier calculates the probability that a detection is true or false based on predictor variables:
- **hit_ratio**: Detection consistency
- **cons_length**: Consecutive detections
- **noise_ratio**: Proportion of miscoded detections
- **power**: Signal strength
- **lag_diff**: Detection timing regularity

See the README for mathematical details.

### What is threshold_ratio?

The classification threshold controls sensitivity:
- **1.0** = Standard (Maximum A Posteriori)
- **>1.0** = Conservative (fewer false positives)
- **<1.0** = Liberal (more detections kept)

Start with 1.0 and adjust based on results.

### How do I know if classification worked?

1. Check training summary histograms - true/false should be well-separated
2. Visualize fish histories - movement should be logical
3. Compare detection counts before/after classification
4. Cross-validate training data

### Can I reclassify data?

Yes! Use `project.reclassify()` with different parameters. Previous classifications are stored with iteration numbers.

### What if I have no training data?

You need at least:
- Beacon tags (known true positives)
- Noise detections (known false positives - automatically identified)

Without training data, MAST cannot classify detections.

## Bouts and Overlap

### Do I need to calculate bouts?

**Recommended but optional.** Bouts are required for:
- Overlap removal (nested doll method)
- Residence time analysis
- Discrete presence/absence modeling

### What is the nested doll algorithm?

An algorithm that removes overlapping detections between receivers with hierarchical detection zones (e.g., large Yagi overlapping small dipole). If a fish is detected at both simultaneously, we keep the more precise location (dipole).

### How do I define overlap relationships?

Create a list of (parent, child) tuples where parent has the larger detection zone:
```python
edges = [('R_yagi', 'R_dipole'), ('R_main', 'R_tributary')]
```

### Can MAST handle complex networks?

Yes! Define all hierarchical relationships. Multiple parents and children are supported.

## Output and Analysis

### What format is the output?

The final recaptures table is available as:
- HDF5 table in the project database
- CSV export (optional)

Both contain all detections with classification results and metadata.

### Can I export to other software?

Yes! MAST can format data for:
- **MARK** (CJS modeling)
- **R** (survival package for competing risks)
- Custom CSV exports

### How do I visualize results?

Use the `fish_history` class to create 3D plots of individual fish movement through the telemetry network.

### Can I use MAST output in R?

Yes! Export recaptures to CSV and import to R. The formatter module creates MARK-compatible files usable in RMark.

## Performance

### MAST is running slowly. What can I do?

1. **Use more RAM** - Close other programs
2. **Process receivers individually** - Don't try to process all at once
3. **Use SSD storage** - HDF5 performs better on SSDs
4. **Optimize bout calculations** - Reduce time_limit parameter
5. **Use parallel processing** - Dask should utilize multiple cores automatically

### How long should processing take?

Depends on data size:
- Import: ~1-5 minutes per receiver
- Training: ~1-10 minutes per receiver
- Classification: ~5-30 minutes per receiver
- Bouts: ~10-60 minutes per node
- Overlap: ~5-30 minutes

Large projects (>1M detections) may take several hours.

## Errors and Troubleshooting

### "No module named 'pymast'"

MAST isn't installed or you're not in the correct environment. Run:
```bash
pip install git+https://github.com/knebiolo/mast.git
```

### "File not found" errors

Check:
1. Project directory path is correct
2. Input CSV files are in the correct location
3. Raw data files are in `Data/Training_Files/`
4. No spaces in file paths (can cause issues)

### Classification produces weird results

1. Check training data quality - view histograms
2. Verify beacon tags are correctly labeled as 'BEACON'
3. Try different predictor combinations
4. Adjust threshold_ratio

### Bout fitting fails

1. Try different time_limit values
2. Manually specify threshold instead of fitting
3. Check if fish are present long enough for bout detection
4. Verify receiver has sufficient detections

### Memory errors

1. Process receivers one at a time
2. Increase available RAM
3. Use `del` to clear large objects
4. Restart Python session between receivers

## Best Practices

### How should I organize my project?

```
MyProject/
├── tblMasterTag.csv
├── tblMasterReceiver.csv
├── tblNodes.csv
├── Data/
│   └── Training_Files/
│       ├── R01_data.txt
│       ├── R02_data.txt
│       └── ...
└── my_analysis_script.py
```

### Should I version control my data?

- **Do** version control: Input CSVs, scripts, configuration
- **Don't** version control: HDF5 database, raw data files, large outputs

### How do I document my analysis?

1. Use configuration files (YAML) for parameters
2. Comment your scripts extensively
3. Keep a lab notebook with decisions made
4. Save training summary figures
5. Document threshold_ratio choices

### What should I include in publications?

- MAST version used
- Predictor variables selected
- Threshold ratio value
- Number of training detections
- Classification performance metrics (if cross-validated)
- Citation to MAST methodology papers

## Contributing

### How can I contribute to MAST?

- Report bugs via GitHub issues
- Suggest features
- Contribute code (parsers, methods, documentation)
- Share example datasets
- Improve documentation

See CONTRIBUTING.md for details.

### I found a bug. What should I do?

1. Check if it's already reported in GitHub issues
2. Create a minimal example that reproduces the bug
3. Open a new issue with:
   - Python version
   - MAST version
   - Error message
   - Minimal code to reproduce

## Getting More Help

### Where can I get support?

1. Read the [Tutorial](TUTORIAL.md)
2. Check the [API Reference](API_REFERENCE.md)
3. Search GitHub issues
4. Open a new issue
5. Contact the maintainers

### Are there example datasets?

Limited examples in the `data/` directory. Full example projects coming soon.

### Can I hire you for consulting?

Contact Kleinschmidt Associates for consulting services related to telemetry projects.

## Citation

### How do I cite MAST?

```
Nebiolo, K.P. and Castro-Santos, T. (2025). MAST: Movement Analysis 
Software for Telemetry. https://github.com/knebiolo/mast
```

See README.md for additional publications to cite.
