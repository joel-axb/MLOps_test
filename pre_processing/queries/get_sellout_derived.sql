SELECT 
    sku, 
    store_id,
    customer_id,
    sellout,
    sellout_raw,
    forecast_dt
from demand_forecast.sellout_derived
where customer_id = 'cosrx' and
    store_id = 'shopify-us' and
    sku in ('SC40AS01', 'SC50AS01', 'SC40TR03')