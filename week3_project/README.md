# E-Commerce Customer Intelligence

Customer segmentation, delivery-delay prediction, and review-score prediction on the Olist Brazilian e-commerce dataset.

For the full technical write-up, see:

- [Report 1](ecommerce_customer_intelligence_report1.pdf)
- [Report 2](ecommerce_customer_intelligence_report2.pdf)
- [Report 3](ecommerce_customer_intelligence_report3.pdf)

## Summary

This project analyzes roughly 100,000 Olist orders from 2016 to 2018 to answer three questions:

- Which customer groups drive revenue?
- What predicts late delivery?
- What predicts low review scores?

### Key results

- RFM segmentation shows a small set of high-value customers contributes a disproportionate share of revenue.
- Review scores drop sharply when deliveries are late.
- The best review-score model is XGBoost, with `0.6238` accuracy on the held-out test set.
- The late-delivery Random Forest model reaches `0.9226` accuracy.

## Project Structure

```text
week3_project/
|-- configs/              # Project configuration
|-- notebooks/            # Ordered analysis notebooks
|-- paper_or_report/      # Full report sources, figures, and tables
|-- reports/              # Generated figures and tables
|-- scripts/              # Download and pipeline scripts
|-- src/                  # Reusable project code
|-- tests/                # Automated tests
|-- website_or_demo/      # Streamlit demo
|-- requirements.txt
|-- environment.yml
|-- Makefile
`-- README.md
```

## Data and Models

- Dataset: [Google Drive folder](https://drive.google.com/drive/folders/1ypxJaQ3a3n-kLiqp0ntxAC8pQEraYWV4?usp=sharing)
- Models: [Google Drive folder](https://drive.google.com/drive/folders/1UOM9GHnlEq0sc-vny0Gz6Ed9OZGf-cNR?usp=sharing)

## Run

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_all.py
streamlit run website_or_demo/app.py
pytest tests -v
```

If you prefer conda:

```bash
conda env create -f environment.yml
conda activate ecommerce_intel
```

## Notes

- The raw dataset is not committed to the repository because of its size.
- The reports contain the full methodology, evaluation, and interpretation details.
