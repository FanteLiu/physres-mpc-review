# PhysRes-MPC Code Release

This repository contains the anonymized code and data used to reproduce the main benchmark and closed-loop control figures for the submitted RA-L manuscript on PhysRes-MPC for phase-change actuator control.

## Contents

- `benchmark/train_data.csv`: identification dataset used for model training and testing.
- `benchmark/model_benchmark.py`: PhysRes, Linear AR, MLP, ESN, and LSTM benchmark script.
- `benchmark/export_paper_benchmark_results.py`: exports benchmark summaries and MATLAB-ready prediction data.
- `benchmark/benchmark_results.md`: summary of benchmark metrics used in the manuscript.
- `benchmark/generate_fig4_fig5_matlab.m`: reproduces pressure and temperature prediction curves.
- `benchmark/generate_fig6_scatter_matlab.m`: reproduces benchmark scatter plots.
- `benchmark/generate_fig7_benchmark_matlab.m`: reproduces accuracy-speed-complexity visualization.
- `closed_loop_control/control_data.mat`: closed-loop pressure tracking data.
- `closed_loop_control/generate_fig8_nature.py`: reproduces the closed-loop control figure.
- `figures/`: reference figure files and generated outputs.

Precomputed files (`benchmark_data.npz`, `benchmark_summary.csv`, and `paper_prediction_data.csv`) are included so that figures can be regenerated without rerunning all models.

## Environment

Python dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The benchmark code uses only lightweight Python dependencies. MATLAB is optional and is used only to reproduce Figs. 4--7 from the exported CSV files.

## Reproduce Benchmark Results

From the repository root:

```bash
cd benchmark
python model_benchmark.py
python export_paper_benchmark_results.py
```

This regenerates:

- `benchmark/benchmark_data.npz`
- `benchmark/benchmark_summary.csv`
- `benchmark/paper_prediction_data.csv`
- `benchmark/benchmark_results.md`

Pressure in `train_data.csv` is stored in hPa. Paper figures and tables report pressure in kPa where indicated.

## Reproduce Paper Figures

### Figs. 4--7

Run the MATLAB scripts from the `benchmark` directory:

```matlab
run('generate_fig4_fig5_matlab.m')
run('generate_fig6_scatter_matlab.m')
run('generate_fig7_benchmark_matlab.m')
```

The generated files are written to `figures/`.

### Fig. 8

From the repository root:

```bash
python closed_loop_control/generate_fig8_nature.py
```

The generated closed-loop control figure is written to `figures/`.

## Notes for Review

This repository is prepared for double-anonymous review. Author names, institutional paths, and local machine paths have been removed from the release files. After review, the repository URL and license information can be updated for the public version.
