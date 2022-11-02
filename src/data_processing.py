import pandas as pd


class FeatureGenerator:
    """Class for joining, extracting and transforming data
    """
    def __init__(
        self, 
        items_df: pd.DataFrame,
        item_categories_df: pd.DataFrame,
        shops_df: pd.DataFrame
    ):
        self.items = items_df
        self.items_cat = item_categories_df
        self.shops = shops_df
    
    def fit(self, sales_df):
        """Generate features
        """
        df = sales_df.copy()
        df = self.__add_items_categories(df)
        df = self.__add_shops(df)
        df = self.__add_shop_location(df)
        df = self.__add_shop_city(df)
        return df
    
    def fit_group(self, sales_df):
        columns = [
            col for col in sales_df.columns 
            if col not in ['item_id', 'shop_id', 'date_block_num', 'item_cnt_day']
        ]
        agg_dict = {col: 'first' for col in columns}
        agg_dict['item_cnt_day'] = 'sum'
        df = sales_df.groupby(by=['item_id', 'shop_id', 'date_block_num']).agg(agg_dict).reset_index()
        df = self.__fill_item_count_zeros(df)
        df = self.fit(df)
        return df
    
    def __add_shops(self, df):
        """add shops information
        """
        return df.join(self.shops, on='shop_id', rsuffix='_r')\
            .drop(columns=['shop_id_r'])

    def __add_shop_city(self, df):
        """Extract city name
        """
        df['shop_city'] = df['shop_name'].apply(lambda x: x.split()[0])
        return df
    
    def __add_shop_location(self, df):
        """Extract shop location
        """
        df['shop_location'] = df['shop_name'].apply(self.__get_location_from_name)
        return df

    def __get_location_from_name(self, x):
        if any([True if i in x else False for i in ['ТЦ', 'ТРК', 'ТРЦ', 'МТРЦ', 'ТК']]):
            return 'shop. center'
        elif any([True if i in x else False for i in ['Онлайн', 'Интернет-магазин']]):
            return 'Online'
        else: return 'other'

    def __add_items_categories(self, df):
        """add items categories information
        """
        item_cat = df.join(self.items, on='item_id', rsuffix='_r')\
            .drop(columns=['item_id_r'])
        item_cat = item_cat.join(self.items_cat, on='item_category_id', rsuffix='_r')\
            .drop(columns=['item_category_id_r'])
        return item_cat
    
    def __fill_item_count_zeros(self, df):
        min_v = df['date_block_num'].min()
        max_v = df['date_block_num'].max()

        df_pivot = pd.pivot(
            df, 
            index=['item_id', 'shop_id'], 
            columns='date_block_num', 
            values='item_cnt_day')
        df_pivot = df_pivot.fillna(0)
        df_long = pd.melt(
            df_pivot, 
            value_vars=[i for i in range(min_v, max_v+1)],
            value_name='item_cnt_day', 
            ignore_index=False
        )
        df_long = df_long.reset_index()
        return df_long


class TrainDataset:
    def __init__(
        self, 
        data: pd.DataFrame, 
        features: list,
        target: str,
        window_size: int,
        start: int
    ):
        self.data = data
        self.features = features
        self.target = target
        self.window = window_size
        self.max_period = self.get_max_period()
        self.start = start
    
    def get_max_period(self):
        return self.data['date_block_num'].max()
    
    def train_validate_split(self, start, end):
        train_data = self.data[
            (self.data['date_block_num'] >= start) & \
            (self.data['date_block_num'] < end)
        ]
        test_data = self.data[self.data['date_block_num']==end]
        train_x, train_y = train_data[self.features], train_data[self.target]
        test_x, test_y = test_data[self.features], test_data[self.target]
        return train_x, test_x, train_y, test_y

    def __iter__(self):
        self.curr_index = self.start
        return self
    
    def __next__(self):
        if self.window + self.curr_index < self.max_period:
            self.curr_index += 1
            return self.train_validate_split(self.curr_index, self.window + self.curr_index)
        raise StopIteration
    
    def __len__(self):
        return self.max_period - self.window - self.start


def clean_sales_data(df, target, percentile):
    df = df.drop_duplicates()
    perc = df[target].quantile(percentile)
    df = df[df[target] <= perc]
    shops = df[df['date_block_num'] > 28]['shop_id'].unique()
    df = df[df['shop_id'].isin(shops)]
    df['shop_item'] = df.apply(lambda x: str(x['item_id']) + '_' + str(x['shop_id']), axis=1)
    items_shops = df[df['date_block_num'] > 22]['shop_item'].unique()
    df = df[df['shop_item'].isin(items_shops)]
    df = df.drop(columns=['shop_item'])
    return df


def group_data(df, by, group):
    df = df.groupby(by=by)
    return df.get_group(group)
