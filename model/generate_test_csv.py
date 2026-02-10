import pandas as pd

def generate_test_csv(path="sample_test_data.csv"):
    data = {
        "Administrative": [0, 1, 2, 3, 4, 1, 5],
        "Administrative_Duration": [0, 45, 80, 120, 180, 60, 240],
        "Informational": [0, 0, 1, 2, 2, 0, 3],
        "Informational_Duration": [0, 0, 30, 90, 120, 0, 150],
        "ProductRelated": [1, 5, 15, 30, 45, 8, 60],
        "ProductRelated_Duration": [30, 300, 600, 1200, 1800, 350, 2500],
        "BounceRates": [0.20, 0.10, 0.05, 0.02, 0.01, 0.12, 0.005],
        "ExitRates": [0.20, 0.15, 0.08, 0.04, 0.02, 0.18, 0.01],
        "PageValues": [0.0, 0.0, 5.2, 18.5, 45.0, 0.0, 80.0],
        "SpecialDay": [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4],
        "Month": [2, 3, 5, 7, 11, 4, 12],
        "OperatingSystems": [1, 2, 2, 3, 2, 1, 3],
        "Browser": [1, 2, 1, 2, 1, 1, 2],
        "Region": [1, 1, 2, 3, 1, 2, 3],
        "TrafficType": [1, 3, 4, 5, 2, 3, 2],
        "VisitorType": [0, 0, 1, 1, 1, 0, 1],  # 0=New, 1=Returning
        "Weekend": [0, 0, 0, 1, 0, 1, 1]
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Sample test CSV saved as: {path}")


if __name__ == "__main__":
    generate_test_csv()
