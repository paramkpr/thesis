# Background

## 2.1 Hierarchical 

### 2.1.1 Definition and Concepts
- Definition of hierarchical clustering and its representation (dendrograms / nested groups on graph)
  - https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html
  - https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/hierarchical-clustering.pdf
  - https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_604
- Distinction between flat clustering and hierarchical clustering
- Agglomerative Clustering Vs. Decisive Clustering
- Key characteristics of hierarchical clustering (nested structure, multi-resolution view)
- Applications in various domains (computational biology, computer imaging, natural language processing)

### 2.1.2 Cost Functions in Hierarchical Clustering (??? where should this go ???)
- Dasgupta's cost function and its significance (clarification needed??)
  - https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/hierarchical-clustering.pdf
  - Dasgupta Cost Paper
  - https://github.com/higra/Higra/tree/master
  

## 2.2 Fairness in Machine Learning

### 2.2.1 Fairness Concepts
- Definition of fairness in machine learning
  - https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_604
  - https://fairmlbook.org/pdf/fairmlbook.pdf
- Individual vs. group fairness
- Real-world examples of algorithmic bias
- Ethical and legal considerations

### 2.2.2 Fair Clustering
- Formulation of fairness constraints in clustering
- Balance constraints and representation guarantees
- Key works in fair flat clustering
- https://www.fairclustering.com/

## 2.3 Fair Hierarchical Clustering

### 2.3.1 Problem Formulation
- Definition of fair hierarchical clustering
- Formal fairness constraints and parameters
- Relative balance in hierarchical structures
- Theoretical challenges specific to hierarchical settings
- https://www.youtube.com/watch?v=wBmyxzXs-ec
- FHC 

### 2.3.2 Algorithmic Approaches
- Key works by Ahmadian et al. and Knittel et al.
  - List of all these algorithms
    - What else?? is there else?? 
    - MakeFair (Knittel, 23)
- Evolution of approximation factors (? don't really get the math ?)
- Tree operators and modification techniques
- Current state-of-the-art algorithms

## 2.4 Theory-Practice Gap in Algorithmic Research
- What is a theory practice gap? 
  - Build from usage in nursing (https://www.sciencedirect.com/science/article/abs/pii/S147159531830204X)
- https://rit.rakuten.com/news/2020/narrowing-the-theory-practice-gap-in-reinforcement-learning-at-this-years-neurips/

### 2.4.1 Theoretical vs. Practical Performance
- Discrepancies between theoretical guarantees and empirical performance
- Assumptions in theoretical analysis and their practical limitations
- Examples of theory-practice gaps in other algorithmic domains

### 2.4.2 Empirical Evaluation Methodologies
- Benchmarking frameworks for algorithmic evaluation
- Dataset considerations and characteristics
- Performance metrics beyond theoretical guarantees
- Challenges in comprehensive empirical evaluation

## 2.5 Benchmarking Systems for Algorithms

### 2.5.1 Existing Benchmarking Approaches
- ...

### 2.5.2 Requirements for Fair Algorithm Benchmarking
- Fairness-specific evaluation metrics
- Balancing fairness and traditional performance metrics
- Transparency and repeatability in benchmark design
- ...