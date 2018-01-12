"""
THE MAIN ALGORITHM FRAME
"""
from process import store_process, clustering

if __name__ == '__main__':
    store_df = store_process()
    store_df = clustering(store_df)
    print(store_df.head())
    store_df.to_csv('store_info.csv', encoding='utf-8', index=False)
