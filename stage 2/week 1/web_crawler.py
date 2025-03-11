import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import os

BOARDS = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]

BASE_URL = "https://www.ptt.cc/bbs/{}/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
COOKIES = {"over18": "1"}
BATCH_SIZE = 1000
TITLE_FOLDER = "ptt_titles"
ID_FOLDER = "ptt_article_ids"

os.makedirs(ID_FOLDER, exist_ok=True)
os.makedirs(TITLE_FOLDER, exist_ok=True)


def fetch_titles(board_name: str) -> None:
    titles = []
    url = BASE_URL.format(board_name)
    print(f"Fetching {board_name} from {url}...")
    last_fetched_id = get_last_fetched_id(board_name)

    latest_article_id = last_fetched_id
    to_continue = True
    while to_continue:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        container = soup.select_one("div.r-list-container.action-bar-margin.bbs-screen")
        articles = []
        for element in container.find_all(recursive=False):
            if "r-list-sep" in element.get("class", []):  # 分隔線
                break
            if "r-ent" in element.get("class", []):
                articles.append(element)

        articles.reverse()  # 由新到舊

        for article in articles:
            link = article.select_one("div.title a")
            if link:
                article_id = extract_article_id(link["href"])
                if should_fetch_article_by_time(article_id, last_fetched_id):
                    title = link.text.strip()
                    titles.append([title])
                    if extract_article_date(article_id) > extract_article_date(
                        latest_article_id
                    ):
                        latest_article_id = article_id
                else:
                    print(f"Stopping {board_name} at {article_id} - already fetched")
                    to_continue = False
                    break

            if len(titles) >= BATCH_SIZE:
                append_to_csv(board_name, titles)
                titles.clear()

        write_latest_fetched_id_to_file(board_name, latest_article_id)

        next_page = get_next_page_url(soup)
        if next_page:
            url = "https://www.ptt.cc" + next_page
        else:
            break

    if titles:
        append_to_csv(board_name, titles)


def extract_article_id(href: str) -> str:
    return href.split("/")[-1].replace(".html", "")


def get_next_page_url(soup: BeautifulSoup) -> str | None:
    link = soup.select_one("a.btn.wide:-soup-contains('‹ 上頁')")

    if link and "disabled" not in link.get("class", []):
        return link.get("href")
    return None


def append_to_csv(board_name: str, titles: list[list[str]]) -> None:
    file_path = os.path.join(TITLE_FOLDER, f"{board_name}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Title"])
        writer.writerows(titles)
    print(f"Appended {len(titles)} new titles to {file_path}")


def write_latest_fetched_id_to_file(board_name: str, article_id: str) -> None:
    file_path = os.path.join(ID_FOLDER, f"{board_name}.txt")
    with open(file_path, "w", encoding="utf-8-sig") as f:
        f.write(f"{article_id}\n")


def get_last_fetched_id(board_name: str) -> str:
    file_path = os.path.join(ID_FOLDER, f"{board_name}.txt")
    if not os.path.isfile(file_path):
        return ""

    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.readline().strip()


def should_fetch_article_by_time(article_id: str, last_fetched_id: str) -> bool:
    last_fetched_time = extract_article_date(last_fetched_id)
    article_time = extract_article_date(article_id)
    return last_fetched_time == "" or article_time > last_fetched_time


def extract_article_date(article_id: str) -> str:
    try:
        timestamp = int(article_id.split(".")[1])
        article_date = datetime.fromtimestamp(timestamp, timezone.utc).strftime(
            "%Y-%m-%d"
        )
        return article_date
    except (IndexError, ValueError):
        return ""


def main() -> None:
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(fetch_titles, BOARDS)


if __name__ == "__main__":
    main()
