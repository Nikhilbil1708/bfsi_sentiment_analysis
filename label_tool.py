# label_tool.py
import csv
import os

OUTPUT_FILE = "data/training_data.csv"
SENTIMENT_OPTIONS = ["positive", "neutral", "negative"]
IMPACT_OPTIONS    = ["bullish",  "neutral", "bearish"]


def label_examples():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "sentiment_label", "impact_label"])

        print("BFSI Sentiment Labeling Tool")
        print("Type the text to label, then choose labels.")
        print("Press Ctrl+C to stop.\n")

        count = 0
        while True:
            try:
                text = input("Text: ").strip()
                if not text:
                    continue

                print(f"Sentiment options: {SENTIMENT_OPTIONS}")
                while True:
                    s = input("Sentiment (p/n/ne): ").strip().lower()
                    sentiment_map = {"p": "positive", "n": "neutral", "ne": "negative",
                                     "positive": "positive", "neutral": "neutral",
                                     "negative": "negative"}
                    if s in sentiment_map:
                        sentiment = sentiment_map[s]
                        break
                    print("Invalid. Use p (positive), n (neutral), ne (negative)")

                print(f"Impact options: {IMPACT_OPTIONS}")
                while True:
                    i = input("Impact (b/n/be): ").strip().lower()
                    impact_map = {"b": "bullish", "n": "neutral", "be": "bearish",
                                  "bullish": "bullish", "neutral": "neutral",
                                  "bearish": "bearish"}
                    if i in impact_map:
                        impact = impact_map[i]
                        break
                    print("Invalid. Use b (bullish), n (neutral), be (bearish)")

                writer.writerow([text, sentiment, impact])
                f.flush()
                count += 1
                print(f"Saved ({count} total). Next example:\n")

            except KeyboardInterrupt:
                print(f"\nDone. {count} examples saved to {OUTPUT_FILE}")
                break


if __name__ == "__main__":
    label_examples()
