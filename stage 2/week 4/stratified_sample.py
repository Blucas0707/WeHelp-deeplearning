import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_sample(
    input_csv: str, output_csv: str, sample_size: int = 10000
) -> None:
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row for row in reader if len(row) >= 2]

    df = pd.DataFrame(data)
    df.columns = ['label'] + [f'token{i}' for i in range(1, df.shape[1])]

    print('🔍 類別分布（原始資料）：')
    print(df['label'].value_counts())

    df_sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df['label'],
        random_state=42,
    )

    print('\n✅ 抽樣後類別分布：')
    print(df_sampled['label'].value_counts())

    df_sampled.to_csv(output_csv, index=False, header=False)
    print(f'✅ 已儲存 {sample_size} 筆 stratified 抽樣資料到：{output_csv}')


if __name__ == '__main__':
    stratified_sample('tokenized_data.csv', 'sample_50k.csv', sample_size=50000)
