with table_1 as (select from_unixtime(cast(order_created_at as bigint)/1000) as order_created_at, item.sku
                from analysis_data.integrated_orders_item
                where customer_id = 'tirtir' and
                        created_at_ym >= '202401')



select date(order_created_at) as order_date, count(*) as count
from table_1
where sku = '01TTF0388' and 
    date(order_created_at) >= date('2024-01-01')
group by 1
order by 1 asc