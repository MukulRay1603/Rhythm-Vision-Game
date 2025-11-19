import pandas as pd
import matplotlib.pyplot as plt


def main():
    try:
        df = pd.read_csv("face_logs.csv")
    except FileNotFoundError:
        print("face_logs.csv not found. Run main.py first.")
        return

    if df.empty:
        print("Log file is empty, nothing to plot.")
        return

    plt.figure(figsize=(7, 6))

    for fid in df["face_id"].unique():
        sub = df[df["face_id"] == fid]
        # x_center, y_center are normalized (0–1)
        plt.plot(sub["x_center"], sub["y_center"], marker=".", linestyle="-", label=f"Face {fid}")

    plt.gca().invert_yaxis()  # image coords: y grows downward
    plt.xlabel("x (normalized)")
    plt.ylabel("y (normalized)")
    plt.title("Face Trajectories (from logs)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
