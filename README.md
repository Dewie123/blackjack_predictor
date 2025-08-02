# Blackjack Predictor using XGBoost

This project implements a machine learning model using **XGBoost** to predict a player's **win percentage** against the dealer's up card in **Blackjack**, and recommends the **optimal action** to take — mimicking the *Perfect Strategy* used by professional players and card counters.

---

## Project Highlights

- **Win Probability Prediction**: Predicts the likelihood of a player's starting hand beating the dealer’s up card.
- **Perfect Strategy Recommender**: Suggests the optimal action (Hit, Stand, Double, etc.) according to professional Blackjack strategy.
- **Data-Driven**: Trained on a simulated dataset of 1 million Blackjack hands, reflecting realistic casino rules and conditions.
- **Model**: Built using the XGBoost classifier for high performance and interpretability.

---

## Demo Video

Watch the YouTube Demo:  
[https://www.youtube.com/watch?v=NimHpAx_hWY](https://www.youtube.com/watch?v=NimHpAx_hWY)

This video shows the model in action, including how it takes input (dealer up card, player hand, true count) and outputs the win percentage and optimal move.

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

2. **(Optional but recommended) Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the predictor**:
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

- **Outputs**:
  - Win probability (float)
  - Recommended move (str): e.g., Hit, Stand, Double, etc.

- **Algorithm**:  
  - [XGBoost](https://xgboost.readthedocs.io/en/stable/) – a powerful gradient boosting decision tree model ideal for tabular data.

---

## Example Output

