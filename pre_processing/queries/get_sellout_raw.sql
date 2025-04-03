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
where customer_id = 'cosrx' and
    store_id = 'shopify-us' and
    sku in ('SC40AS01', 'SC50AS01', 'SC40TR03')