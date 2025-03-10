# Thesis Introduction Writing Guide

## Past theses intro section structure

1. **Opening context and motivation** (1-2 paragraphs)
2. **Problem statement** (1-2 paragraphs)
3. **Research approach overview** (1-2 paragraphs)
4. **Specific contribution statement** (1-2 paragraphs)
5. **Paper organization/roadmap** (optional, sometimes included at end)

## Possible Introduction Components

### 1. Opening Context & Motivation
- Begin with a broad statement about the gap between theoretical algorithmic guarantees and practical implementation
- Mention specific examples (fair hierarchical clustering / massively parallel whichever way we go)
- Establish why this matters (real-world impact, computational efficiency concerns)

### 2. Problem Statement
- Clearly articulate the research question: "How do theoretical guarantees in [set of algorithms] translate to practical performance?"
- Define the specific challenges:
  - Identify algorithms with theory-practice gaps
  - Lack of standardized benchmarking methodologies
  - Absence of comprehensive empirical studies
- Emphasize the gap in the literature (similar to how Austin's thesis mentions "PAT+ is the first to evaluate the security of generated code")

### 3. Research Approach
- Detail your methodological framework:
  - Literature review to identify algorithmic domains with theory-practice gaps
  - #### Lit Review Section
    - A brief literature review section (typically 1-2 paragraphs) should:
      - Acknowledge seminal work in algorithm analysis and empirical evaluation
      - Identify the specific disconnect between theory and practice in your focus areas
      - Note previous attempts at bridging this gap and their limitations
      - Signal to the reader that a more comprehensive review appears in your background chapter

    ##### Example
    - Austin's thesis references existing work (PAT+) and positions his research as building upon it
    - Madison's thesis discusses consensus algorithms broadly before narrowing to fairness
    - Both theses introduce key concepts and cite foundational papers to orient the reader
  - Development of benchmarking environment
  - Empirical analysis using synthetic and real-world datasets
  - Include a simple illustration or diagram if appropriate (both sample theses use figures)

### 4. Contribution Statement
- Articulate specifically what your research contributes:
  - "In this paper, we provide the first comprehensive empirical study of..."
  - Benchmarking framework as a foundation for future work
  - Evidence-based algorithm selection insights


## Possible Intro Structure
[Opening context about algorithm theory
[Specific example of fair hierarchical clustering/parallel (idk placeholder for now) and other algorithms]
[Problem statement about lack of empirical evaluation frameworks]
[Figure illustrating the gap or your proposed framework]
[Your research approach in 3 parts: literature review, benchmarking, empirical analysis]
[Specific contributions of your work]
[Optional: Brief chapter overview]

## Notes

- Aim for 2-3 pages for the introduction (both)
- Balance technical detail with accessibility
- Center your unique contribution in the algorithmic evaluation space
- Make connections to both theoretical computer science and practical applications
