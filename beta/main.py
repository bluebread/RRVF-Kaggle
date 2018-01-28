"""
主要算法架构
"""
from process import store_process

if __name__ == '__main__':
    store_df = store_process()
    print(store_df.head())
    store_df.to_csv('store_info.csv', encoding='utf-8', index=False)
