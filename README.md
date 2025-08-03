# Blackjack Predictor using XGBoost

This project implements a machine learning model using **XGBoost** to predict a player's **win percentage** against the dealer's up card in **Blackjack**, and recommends the **optimal action** to take , replicating the *Perfect Strategy* used by professional players and card counters.

---


## Demo Video

Watch the YouTube Demo:  
[https://www.youtube.com/watch?v=NimHpAx_hWY](https://www.youtube.com/watch?v=NimHpAx_hWY)

This video shows the model in action, including how it takes input (dealer up card, player hand, true count) and outputs the win percentage and optimal move.

---


## Project Highlights

- **Win Probability Prediction**: Predicts the likelihood of a player's starting hand beating the dealer’s up card.
- **Perfect Strategy Recommender**: Suggests the optimal action (Hit, Stand, Double, etc.) according to professional Blackjack strategy.
- **Data-Driven**: Trained on a simulated dataset of 1 million Blackjack hands, reflecting realistic casino rules and conditions.
- **Model**: Built using the XGBoost classifier for high performance and interpretability.

---

### Data Exploration and Preprocessing Strategy

A comprehensive exploratory data analysis (EDA) was performed to rigorously characterize the statistical properties and interdependencies within the dataset. Key observations informed a deliberately minimalistic preprocessing approach tailored to the unique characteristics of blackjack gameplay data:

- The dataset contains critical continuous features such as **True Count** and **Shoe Number**, which encapsulate temporal and contextual information fundamental to strategic decision-making in blackjack.
- Conventional data balancing techniques such as **SMOTE** or synthetic oversampling, while effective in many classification tasks, were deemed unsuitable here. Applying such methods would artificially interpolate new samples, thereby **disrupting inherent feature correlations and sequence-dependent structures**, and introducing spurious patterns inconsistent with actual gameplay dynamics.
- Maintaining the integrity of the raw data distribution was essential to preserve the fidelity of the feature interactions and avoid biasing the model with synthetic artifacts.

This approach highlights the critical need to preserve the intrinsic structure of blackjack gameplay data such as card count sequences and true count dynamics when designing preprocessing steps, since disrupting these domain-specific correlations would compromise the model’s ability to accurately replicate the mathematically precise strategies fundamental to optimal blackjack play.

### Model Evaluation and Rationale for Choosing XGBoost

Multiple supervised learning algorithms were evaluated to identify the optimal predictive framework capable of modeling the complex, nonlinear relationships inherent in blackjack strategy optimization:

- **Random Forests**: Ensemble tree-based methods offering robustness and interpretability. In testing, Random Forests showed limitations in capturing subtle interaction effects between continuous features with sufficient granularity, and exhibited longer training times.
  
- **Deep Neural Networks (DNNs)**: Although DNNs excel at modeling complex feature hierarchies, their application here presents challenges:
  - Necessitate extensive hyperparameter tuning and risk overfitting due to the relatively low dimensionality and structured nature of the features.
  - Overcomplication due to fine-tuning activation function (i used elu for this), for minor improvements in accuracy which is irrelevant in this example due to the hard ceiling of model performance(Mathematically-proven perfect strategy)

- **XGBoost (Extreme Gradient Boosting)**: This gradient-boosted decision tree framework emerged as the superior choice, delivering an optimal balance of **predictive performance, computational efficiency, and explainability**. Its advantages include:
  - Sophisticated regularization mechanisms mitigating overfitting.
  - Ability to inherently handle mixed continuous and categorical variables.
  - Fast training leveraging parallel and distributed computing.
  - Built-in tools for feature importance and SHAP value explanations, enabling clear insight into model decision processes.

Critically, the finalized XGBoost model attained a **remarkable 98% efficacy**, effectively replicating the mathematically proven **Perfect Strategy** employed by expert blackjack players and card counters. This near-perfect alignment confirms the model’s capacity to generalize domain rules without overfitting or reliance on excessive complexity.

Consequently, the marginal gains from deploying a deep neural architecture do not justify the added complexity, opacity, and resource demands. The XGBoost solution represents a **best-in-class synthesis of accuracy, robustness, and transparency**, perfectly suited for this specialized application.


---


## Dataset

- **Source**: [Blackjack Hands on Kaggle](https://www.kaggle.com/datasets/dennisho/blackjack-hands)
- **Filename**: `blackjack_simulator.csv`

### Dataset Details
Simulated using a realistic blackjack engine based on standard Las Vegas rules:
- 8-deck shoe (6.5 deck penetration)
- Dealer hits soft 17
- Blackjack pays 3:2
- Double down allowed on any first 2 cards
- Double after split allowed
- Split same cards up to 4 hands
- No resplitting Aces
- Aces receive one extra card only (no blackjack)
- Late surrender allowed (not after split)

### Card Representation
- 10s, Jacks, Queens, Kings → `10`
- Aces → `11` (always, regardless of soft/hard usage)
- Suits are not recorded
- True count and run count recorded using Hi-Lo system
- Actions taken:
  - `H`: Hit  
  - `S`: Stand  
  - `D`: Double Down  
  - `P`: Split  
  - `R`: Surrender  
  - `I`: Buy Insurance (never used)  
  - `N`: No Insurance


---


## Getting Started

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/blackjack_predictor_XGBoost.git
    cd blackjack_predictor_XGBoost
    ```

2. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the predictor**:
    ```bash
    python blackjack_predictor.py
    ```

---

## Model Overview

- **Inputs**:
  - Dealer up card
  - Player's first card
  - Player's second card
  - True count
  - Run count
  - Cards remaining in shoe

- **Outputs**:
  - Win probability (float)
  - Recommended move (str): e.g., Hit, Stand, Double, etc.

- **Algorithm**:  
  - [XGBoost](https://xgboost.readthedocs.io/en/stable/)


---


## Example Output
![Example output](assets/Example_Output.PNG)

## Multi Deck Perfect Strategy - Mathematically derived optimal move
![Multi Deck Perfect Strategy](assets/Multi_Deck_Perfect_Strategy_Hit_Soft_17.PNG)
