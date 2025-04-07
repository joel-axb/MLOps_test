SELECT 
    sku, 
    store_id,
    customer_id,
    forecast_dt,
    promotion_target_day,
    promotion_day_type,
    discount_rate
FROM demand_forecast.sellout_timeseries
where customer_id in {customer_id} and
    store_id in {store_id}