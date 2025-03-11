fair_hierarchical_clustering/
├── data/
│   ├── __init__.py
│   ├── synthetic.py        # Generate synthetic clustering data                            [*]
│   └── real_datasets.py    # Load and process real datasets (adult.csv, bank.csv)          [ ]
├── models/
│   ├── __init__.py
│   ├── node.py             # Node class for hierarchical tree representation               [*]
│   └── hierarchy.py        # Tree structure and core operations                            [*]
├── algorithms/
│   ├── __init__.py
│   ├── average_linkage.py  # Vanilla hierarchical clustering                               [*]
│   ├── split_root.py       # SplitRoot algorithm                                           [*]
│   └── make_fair.py        # MakeFair algorithm                                            [~]    
├── experiments/
│   ├── __init__.py
│   ├── config.py           # Configuration for experiments                                 [ ]    
│   ├── runner.py           # Run experiments with different parameters                     [ ]
│   └── metrics.py          # Evaluate fairness and cost metrics                            [ ]
├── visualization/
│   ├── __init__.py
│   ├── dendrograms.py      # Visualize hierarchical trees                                  [ ]
│   ├── balance_plots.py    # Plot cluster balance distributions                            [ ]
│   └── cost_plots.py       # Plot cost ratios and other metrics                            [ ]
├── utils/
│   ├── __init__.py
│   ├── distance.py         # Distance and similarity calculations                          [*]
│   └── tree_operations.py  # Common tree manipulation operations                           [~]
├── tests/                  # Unit tests                                                                
│   ├── __init__.py                                                                         [*]
│   ├── test_node.py
│   ├── test_algorithms.py
│   └── test_metrics.py
└── main.py                 # Entry point for running experiments