"""Habit and Mood Tracker - single file version
Needs only:  pip install numpy matplotlib
Run:         python habit_tracker_single.py
"""
import csv
import os
import tkinter as tk
from datetime import date
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# PART 1: DailyLog class and CSV storage  (Unit 1: classes, file handling)
# =====================================================================
FILE_NAME = "habit_data.csv"
FIELDS = ["date", "sleep", "study", "screen", "mood"]


class DailyLog:
    """Represents one day's habits and mood."""

    def __init__(self, day, sleep, study, screen, mood):
        self.day = day
        self.sleep = float(sleep)    # hours slept
        self.study = float(study)    # hours studied
        self.screen = float(screen)  # hours of screen time
        self.mood = float(mood)      # 1 to 10

    def to_row(self):
        return [self.day, self.sleep, self.study, self.screen, self.mood]


def save_log(log):
    """Append one DailyLog to the CSV file (creates file with header if new)."""
    new_file = not os.path.exists(FILE_NAME)
    with open(FILE_NAME, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow(log.to_row())


def load_logs():
    """Read all entries from the CSV file and return a list of DailyLog objects."""
    logs = []
    if not os.path.exists(FILE_NAME):
        return logs
    with open(FILE_NAME, newline="") as f:
        for row in csv.DictReader(f):
            logs.append(DailyLog(row["date"], row["sleep"], row["study"],
                                 row["screen"], row["mood"]))
    return logs


# =====================================================================
# PART 2: NumPy calculations  (Unit 2: ndarray, statistical functions)
# =====================================================================
HABITS = ["sleep", "study", "screen"]


def to_arrays(logs):
    """Convert a list of DailyLog objects into NumPy arrays."""
    return {
        "sleep": np.array([l.sleep for l in logs]),
        "study": np.array([l.study for l in logs]),
        "screen": np.array([l.screen for l in logs]),
        "mood": np.array([l.mood for l in logs]),
    }


def weekly_summary(logs):
    """Average of the last 7 entries, plus best and worst mood day."""
    last7 = logs[-7:]
    data = to_arrays(last7)
    best = last7[int(np.argmax(data["mood"]))]
    worst = last7[int(np.argmin(data["mood"]))]
    return {
        "entries": len(last7),
        "avg_sleep": round(float(np.mean(data["sleep"])), 2),
        "avg_study": round(float(np.mean(data["study"])), 2),
        "avg_screen": round(float(np.mean(data["screen"])), 2),
        "avg_mood": round(float(np.mean(data["mood"])), 2),
        "best_day": best.day,
        "worst_day": worst.day,
    }


def correlation_matrix(logs):
    """Correlation between all habits and mood (values from -1 to 1)."""
    data = to_arrays(logs)
    labels = HABITS + ["mood"]
    stacked = np.array([data[k] for k in labels])
    matrix = np.corrcoef(stacked)
    return np.nan_to_num(matrix), labels  # nan appears if a column never changes


def biggest_influence(logs):
    """Find which habit has the strongest link with mood."""
    matrix, labels = correlation_matrix(logs)
    mood_row = matrix[-1][:-1]  # mood vs each habit
    idx = int(np.argmax(np.abs(mood_row)))
    return labels[idx], round(float(mood_row[idx]), 2)


# =====================================================================
# PART 3: Charts  (Unit 2: matplotlib)
# =====================================================================
def mood_line_plot(logs):
    data = to_arrays(logs)
    days = [l.day for l in logs]
    plt.figure(figsize=(8, 4))
    plt.plot(days, data["mood"], marker="o", color="teal")
    plt.title("Mood Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mood (1-10)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def sleep_vs_mood_scatter(logs):
    data = to_arrays(logs)
    plt.figure(figsize=(6, 5))
    plt.scatter(data["sleep"], data["mood"], color="purple", s=60)
    if len(logs) >= 2 and np.std(data["sleep"]) > 0:
        slope, intercept = np.polyfit(data["sleep"], data["mood"], 1)
        x = np.linspace(data["sleep"].min(), data["sleep"].max(), 50)
        plt.plot(x, slope * x + intercept, "r--", label="Trend")
        plt.legend()
    plt.title("Sleep vs Mood")
    plt.xlabel("Sleep (hours)")
    plt.ylabel("Mood (1-10)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def mood_histogram(logs):
    data = to_arrays(logs)
    plt.figure(figsize=(6, 4))
    plt.hist(data["mood"], bins=range(1, 12), color="orange", edgecolor="black")
    plt.title("Mood Distribution")
    plt.xlabel("Mood score")
    plt.ylabel("Number of days")
    plt.tight_layout()
    plt.show()


def habit_heatmap(logs):
    matrix, labels = correlation_matrix(logs)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center")
    plt.title("Which Habit Affects Mood Most?")
    plt.tight_layout()
    plt.show()


# =====================================================================
# PART 4: Tkinter GUI  (Unit 3: widgets, events, geometry managers)
# =====================================================================
class HabitApp:
    def __init__(self, root):
        self.root = root
        root.title("Habit and Mood Tracker")
        root.geometry("420x540")

        tk.Label(root, text="Habit and Mood Tracker",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # ---- Entry form (grid geometry manager) ----
        form = tk.Frame(root)
        form.pack(pady=5)

        self.entries = {}
        fields = [
            ("Date (YYYY-MM-DD)", "day"),
            ("Sleep (hours)", "sleep"),
            ("Study (hours)", "study"),
            ("Screen time (hours)", "screen"),
            ("Mood (1-10)", "mood"),
        ]
        for row, (label, key) in enumerate(fields):
            tk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = tk.Entry(form, width=18)
            entry.grid(row=row, column=1, padx=10)
            self.entries[key] = entry
        self.entries["day"].insert(0, str(date.today()))

        # Pressing Enter in the mood box also saves (key event binding)
        self.entries["mood"].bind("<Return>", lambda e: self.save_entry())

        tk.Button(root, text="Save Entry", bg="#4CAF50", fg="white",
                  width=20, command=self.save_entry).pack(pady=10)

        # ---- Analysis buttons ----
        tk.Button(root, text="Weekly Summary", width=24,
                  command=self.show_summary).pack(pady=3)
        tk.Button(root, text="Mood Over Time (Line)", width=24,
                  command=lambda: self.show_chart(mood_line_plot)).pack(pady=3)
        tk.Button(root, text="Sleep vs Mood (Scatter)", width=24,
                  command=lambda: self.show_chart(sleep_vs_mood_scatter)).pack(pady=3)
        tk.Button(root, text="Mood Distribution (Histogram)", width=24,
                  command=lambda: self.show_chart(mood_histogram)).pack(pady=3)
        tk.Button(root, text="Habit Heatmap", width=24,
                  command=lambda: self.show_chart(habit_heatmap)).pack(pady=3)

        # ---- Result label ----
        self.result = tk.Label(root, text="", justify="left", wraplength=380,
                               fg="#333")
        self.result.pack(pady=10)

    def save_entry(self):
        """Validate input and save one day's entry."""
        try:
            day = self.entries["day"].get().strip()
            sleep = float(self.entries["sleep"].get())
            study = float(self.entries["study"].get())
            screen = float(self.entries["screen"].get())
            mood = float(self.entries["mood"].get())
        except ValueError:
            messagebox.showerror("Error", "Please enter numbers in all fields.")
            return

        if not day:
            messagebox.showerror("Error", "Please enter a date.")
            return
        if not (1 <= mood <= 10):
            messagebox.showerror("Error", "Mood must be between 1 and 10.")
            return
        if not all(0 <= h <= 24 for h in (sleep, study, screen)):
            messagebox.showerror("Error", "Hours must be between 0 and 24.")
            return

        save_log(DailyLog(day, sleep, study, screen, mood))
        messagebox.showinfo("Saved", "Entry saved successfully!")

        # Clear the number fields for the next entry
        for key in ("sleep", "study", "screen", "mood"):
            self.entries[key].delete(0, tk.END)

    def show_summary(self):
        """Show weekly averages and the biggest influence on mood."""
        logs = load_logs()
        if not logs:
            messagebox.showwarning("No data", "Add at least one entry first.")
            return
        s = weekly_summary(logs)
        text = (f"Last {s['entries']} entries\n"
                f"Avg sleep: {s['avg_sleep']} h\n"
                f"Avg study: {s['avg_study']} h\n"
                f"Avg screen time: {s['avg_screen']} h\n"
                f"Avg mood: {s['avg_mood']} / 10\n"
                f"Best day: {s['best_day']}   Worst day: {s['worst_day']}")
        if len(logs) >= 3:
            habit, value = biggest_influence(logs)
            text += f"\n\nBiggest influence on mood: {habit} (correlation {value})"
        self.result.config(text=text)

    def show_chart(self, chart_function):
        """Load data and call the given chart function."""
        logs = load_logs()
        if len(logs) < 3:
            messagebox.showwarning("Not enough data",
                                   "Add at least 3 entries to see charts.")
            return
        chart_function(logs)


if __name__ == "__main__":
    root = tk.Tk()
    HabitApp(root)
    root.mainloop()