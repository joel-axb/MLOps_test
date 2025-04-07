SELECT 
    sku, 
    store_id,
    customer_id,
    sellout,
    sellout_raw,
    forecast_dt
from demand_forecast.sellout_derived
where customer_id in {customer_id} and
    store_id in {store_id}