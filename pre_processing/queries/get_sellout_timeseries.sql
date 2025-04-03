SELECT 
    sku, 
    store_id,
    customer_id,
    forecast_dt,
    promotion_target_day,
    promotion_day_type,
    discount_rate
FROM demand_forecast.sellout_timeseries
where customer_id = 'cosrx' and
    store_id = 'shopify-us' and
    sku in ('SC40AS01', 'SC50AS01', 'SC40TR03')