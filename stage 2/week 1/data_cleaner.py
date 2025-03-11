import os
import pandas as pd
import glob
import re

INPUT_FOLDER = "ptt_titles"
OUTPUT_FILE = "cleaned_data.csv"


# 定義標題清理函式
def clean_title(title: str) -> str:
    if not isinstance(title, str):
        return None

    title = title.strip().lower()  # 去空格 & 轉小寫
    # 移除 Re: & FW:
    if title.startswith("re:") or title.startswith("fw:"):
        return None

    if not re.match(r"^\[.*?\]", title):
        return None  # 移除沒有分類標籤的標題

    if any(keyword in title for keyword in ["公告", "水桶", "警告"]):
        return None  # 移除包含「公告」、「水桶」或「警告」的標題

    title = re.sub(r"http\S+|www\S+", "", title)  # 移除網址
    title = re.sub(r"[^\w\s#@]", "", title)  # 移除特殊符號，保留 # 和 @

    # 移除過短標題
    if len(title) < 3:
        return None

    return title


def clean_data():
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    cleaned_data = []

    for file in all_files:
        board_name = os.path.splitext(os.path.basename(file))[0]
        print(f"Start cleaning {board_name} ...")
        df = pd.read_csv(file, header=None, names=["title"])

        df["title"] = df["title"].apply(clean_title)
        df.dropna(subset=["title"], inplace=True)

        for title in df["title"]:
            cleaned_data.append([board_name, title])

    cleaned_df = pd.DataFrame(cleaned_data, columns=["board_name", "title"])
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_data()
