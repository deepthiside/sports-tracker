"""
Plot active subject count over time.
Usage: python count_plot.py
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_count(csv: str = "output/count_over_time.csv",
               output: str = "output/count_over_time.png"):
    df = pd.read_csv(csv)
    plt.figure(figsize=(14, 5))
    plt.plot(df["frame"], df["count"], color="royalblue", linewidth=1.5)
    plt.fill_between(df["frame"], df["count"], alpha=0.2, color="royalblue")
    plt.xlabel("Frame Number")
    plt.ylabel("Active Subjects")
    plt.title("Active Tracked Subjects Over Time")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"[OUT] Count plot → {output}")
    plt.show()

if __name__ == "__main__":
    plot_count()