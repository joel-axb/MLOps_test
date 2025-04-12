with table_1 as (select from_unixtime(cast(order_created_at as bigint)/1000) as order_created_at, item.sku
                from analysis_data.integrated_orders_item
                where customer_id in {customer_id} and
                        created_at_ym >= '202503' and
                        sstore_id in {store_id})



select date(order_created_at) as order_date, sku, count(*) as count
from table_1
where date(order_created_at) >= date('2025-03-01')
group by 1, 2
order by 1 asc