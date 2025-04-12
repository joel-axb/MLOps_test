SELECT DISTINCT customer_id, store_id, ssku
FROM "raw_data"."products_store_item" 
where customer_id in {customer_id} and
    store_id in {store_id}