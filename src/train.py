from pprint import pprint

from src.pipeline import train_pipeline


def train() -> None:
    metrics = train_pipeline()
    print("\nTraining complete. Summary:")
    pprint(metrics["metadata"])
    print("\nClassification report:")
    pprint(metrics["classification_report"])


if __name__ == "__main__":
    train()
