import pandas as pd
import joblib 


win_model = joblib.load("win_model.pkl")
action_model = joblib.load("action_model.pkl")
le = joblib.load("label_encoder.pkl")


def predict_blackjack_strategy(dealer_up, initial_hand, true_count, run_count, cards_remaining):
    card1, card2 = initial_hand
    is_pair = int(card1 == card2)
    has_ace = int(11 in initial_hand)
    hand_sum = sum(initial_hand)
    is_blackjack = int(hand_sum == 21)

    sample = pd.DataFrame([{
        'dealer_up': dealer_up,
        'card1': card1,
        'card2': card2,
        'is_pair': is_pair,
        'has_ace': has_ace,
        'hand_sum': hand_sum,
        'true_count': true_count,
        'run_count': run_count,
        'cards_remaining': cards_remaining,
        'is_blackjack': is_blackjack
    }])

    win_prob = win_model.predict_proba(sample)[0][1]
    action_encoded = action_model.predict(sample)[0]
    action = le.inverse_transform([action_encoded])[0]

    return round(win_prob * 100, 2), action

def switch(action):
        if action == 'H':
            return 'Hit'
        elif action == 'S':
            return 'Stand'
        elif action == 'D':
            return 'Double Down'
        elif action == 'P':
            return 'Split'
        elif action == 'R':
            return 'Surrender, if casino does not allow surrender, Stand instead.'
        elif action == 'I':
            return 'Buy Insurance'
        elif action == 'N':
            return 'No Insurance'

if __name__ == "__main__":

    while True:

        print("Please enter the following parameters for blackjack prediction:")

        dealer_up = int(input("Dealer's up card (2-11): "))
        card1 = int(input("Your first card (2-11): "))
        card2 = int(input("Your second card (2-11): "))
        true_count = float(input("True count (e.g. 5): "))
        run_count = int(input("Run count (e.g. 1): "))
        cards_remaining = int(input("Cards remaining in the shoe (e.g. 416): "))

        initial_hand = [card1, card2]

        win_percent, action = predict_blackjack_strategy(
            dealer_up=dealer_up,
            initial_hand=initial_hand,
            true_count=true_count,
            run_count=run_count,
            cards_remaining=cards_remaining
        )

        

        print("\nPrediction Results:")
        print(f"Estimated Win %: {win_percent}%")
        print(f"Recommended Action: {switch(action)}")

        user_input = input("\nWould you like to continue? [Y/N]: ").strip().upper()

        if user_input ==  'N':
            break

