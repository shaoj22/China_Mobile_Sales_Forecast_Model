

import csv
import random
from datetime import datetime, timedelta


def generate_train_csv(years, num_stores, num_items, sales_range, name):
    start_date = datetime(2013, 1, 1)
    end_date = start_date + timedelta(days=years * 365)
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['date', 'store', 'item', 'sales']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for store in range(1, num_stores + 1):
            for item in range(1, num_items + 1):
                current_date = start_date
                while current_date < end_date:
                    sales = random.randint(sales_range[0], sales_range[1])
                    writer.writerow({'date': current_date.strftime('%Y-%m-%d'), 'store': store, 'item': item, 'sales': sales})
                    current_date += timedelta(days=1)

# 示例用法
generate_train_csv(years=7, num_stores=10, num_items=10, sales_range=[1, 100], name="trainInstance1.csv")