# Datasheet for the Olist Brazilian E-Commerce Dataset

Based on the Datasheets for Datasets framework (Gebru et al., 2021).

## Motivation

**Why was this dataset created?**
Olist released this dataset to encourage data science research on Brazilian
e-commerce.  It was published on Kaggle in 2018 as a fully anonymised
snapshot of the Olist Store marketplace.

**Who funded it?**
Olist (https://olist.com).

## Composition

| Property | Value |
|----------|-------|
| Total orders | ~100,000 |
| Date range | Sep 2016 – Oct 2018 |
| Delivered orders | ~96,000 |
| Unique customers | ~99,000 |
| Unique sellers | ~3,000 |
| Product categories | ~70 |
| Brazilian states covered | 27 |

**Tables:**
- olist_orders_dataset.csv (99,441 rows)
- olist_order_items_dataset.csv (112,650 rows)
- olist_order_payments_dataset.csv (103,886 rows)
- olist_order_reviews_dataset.csv (99,224 rows)
- olist_customers_dataset.csv (99,441 rows)
- olist_products_dataset.csv (32,951 rows)
- olist_sellers_dataset.csv (3,095 rows)
- olist_geolocation_dataset.csv (1,000,163 rows)
- product_category_name_translation.csv (71 rows)

## Collection Process

Data was collected from the Olist Store's production database.
Personally identifiable information was anonymised: customer IDs and seller
IDs are random hashes; review comment text is included but does not contain
names or contact details.

## Preprocessing / Cleaning Done by Olist

- Customer and seller identifiers are anonymised.
- Orders are linked to customers via a `customer_id` + `customer_unique_id`
  design (one customer can appear multiple times with different `customer_id` values
  across orders, but with the same `customer_unique_id`).

## Uses

**Intended uses:**
- E-commerce analytics research
- Customer segmentation
- Delivery performance analysis
- Natural language processing on review text

**Uses to avoid:**
- Re-identification of customers or sellers
- Commercial use without checking CC BY-NC-SA 4.0 licence terms

## Distribution

- License: CC BY-NC-SA 4.0
- URL: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- DOI / Citation: See references.bib

## Maintenance

Maintained by Olist.  The dataset represents a historical snapshot and is
not regularly updated.

## Known Issues

1. Review comments contain some Portuguese text that may require translation.
2. A small number of orders have inconsistent timestamps (approved before purchase).
3. Geolocation data has more rows than orders (it is a ZIP-level lookup table,
   not order-level).
4. Some product weights and dimensions are missing or implausible.
