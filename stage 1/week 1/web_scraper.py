import json
import ssl
from math import sqrt
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen


BASE_URL = 'https://ecshweb.pchome.com.tw/search/v4.3/all/results'
ASUS_CATE_ID = 'DSAA31'
ITEM_COUNT_PER_PAGE = 40
PAGE = 1


class WebScaper:
    @staticmethod
    def scrape_web(url: str) -> str:
        # handle SSL issue
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        try:
            with urlopen(url, context=context) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            raise RuntimeError('Fail to scrape web content') from e


class ProductDataFetcher:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def fetch_all_product_ds(
        self,
        cate_id: str = None,
        attr: str = None,
        item_count_per_page: int = ITEM_COUNT_PER_PAGE,
        page: int = PAGE,
    ) -> list[dict]:
        product_ds = []
        query_params = {
            'cateid': cate_id,
            'attr': attr,
            'pageCount': item_count_per_page,
            'page': page,
        }

        while True:
            query_string = urlencode(query_params)
            url = urljoin(self.base_url, f'?{query_string}')

            result_d = json.loads(WebScaper.scrape_web(url))

            prod_ds = result_d.get('Prods')
            if not prod_ds:
                break

            product_ds += prod_ds
            query_params['page'] += 1

        return product_ds


class FileUtil:
    @staticmethod
    def write_to_file(filename: str, lines: list[str]) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class ProductAnalyzer:
    @staticmethod
    def calculate_average_price(prod_ds: list[dict]) -> float:
        if not prod_ds:
            return 0.0

        total_price = sum(prod_d['Price'] for prod_d in prod_ds)
        return total_price / len(prod_ds)

    @staticmethod
    def filter_best_product_ids(
        prod_ds: list[dict], gt_rating: float = 4.9, gt_reviews: int = 0
    ) -> list[str]:
        return [
            prod_d['Id']
            for prod_d in prod_ds
            if prod_d.get('ratingValue') is not None
            and prod_d.get('ratingValue') > gt_rating
            and prod_d.get('reviewCount') is not None
            and prod_d.get('reviewCount') >= gt_reviews
        ]

    @staticmethod
    def calculate_z_scores(prod_ds: list[dict], average_price: float) -> list[str]:
        if not prod_ds:
            return []

        price_diff_sum = sum(
            (prod_d['Price'] - average_price) ** 2 for prod_d in prod_ds
        )
        sigma = sqrt(price_diff_sum / len(prod_ds))
        return [
            f'{prod_d["Id"]},{prod_d["Price"]},{(prod_d["Price"] - average_price) / sigma:.2f}'
            for prod_d in prod_ds
        ]


class TaskHandler:
    def __init__(self, fetcher: ProductDataFetcher) -> None:
        self.fetcher = fetcher

    def task_1_scrape_asus_items(self) -> None:
        prod_ds = self.fetcher.fetch_all_product_ds(ASUS_CATE_ID)
        FileUtil.write_to_file('products.txt', [prod_d['Id'] for prod_d in prod_ds])

    def task_2_scrape_best_products(self) -> None:
        prod_ds = self.fetcher.fetch_all_product_ds(ASUS_CATE_ID)
        best_prod_ids = ProductAnalyzer.filter_best_product_ids(prod_ds)
        FileUtil.write_to_file('best-products.txt', best_prod_ids)

    def task_3_calculate_average_price_for_intel_i5(self) -> None:
        prod_ds = self.fetcher.fetch_all_product_ds(ASUS_CATE_ID, attr='G26I2272')
        average_price = ProductAnalyzer.calculate_average_price(prod_ds)
        print(f'{average_price:.1f}')

    def task_4_calculate_z_scores(self) -> None:
        prod_ds = self.fetcher.fetch_all_product_ds(ASUS_CATE_ID)
        average_price = ProductAnalyzer.calculate_average_price(prod_ds)
        z_scores = ProductAnalyzer.calculate_z_scores(prod_ds, average_price)
        FileUtil.write_to_file('standardization.csv', z_scores)


if __name__ == '__main__':
    fetcher = ProductDataFetcher(base_url=BASE_URL)
    task_handler = TaskHandler(fetcher)

    task_handler.task_1_scrape_asus_items()
    task_handler.task_2_scrape_best_products()
    task_handler.task_3_calculate_average_price_for_intel_i5()
    task_handler.task_4_calculate_z_scores()
