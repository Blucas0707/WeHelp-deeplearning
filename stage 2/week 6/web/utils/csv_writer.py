import csv


def save_to_csv(title: str, label: str, filename='user-labeled-titles.csv'):
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([title, label])
