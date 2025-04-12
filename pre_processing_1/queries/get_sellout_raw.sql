SELECT 
    sku,
    max_price,
    today_price,
    discount_value_skudiscount,
    discount_rate_skudiscount,
    discount_value_alldiscount,
    gift_price,
    customer_id,
    store_id,
    forecast_dt
FROM demand_forecast.sellout_raw
where customer_id in {customer_id} and
    store_id in {store_id}