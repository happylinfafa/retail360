# Retail360 three-minute demo transcript

The video records a local run of the same Streamlit code in this repository, using stored historical customer examples. Narration uses the Windows Microsoft Zira synthetic voice. No paid API was used.

## 0:00–0:30

Hello. This is Retail360, a research prototype for retail analysts. The main question is simple: using a customer's past orders, can we predict whether that customer will buy again in the next sixty days? A retailer with limited time can use a ranking to decide which customer records to review first. The prediction supports a human decision; it is not an automatic marketing action.

## 0:30–1:00

We use the public UCI Online Retail Two dataset. At each prediction date, we look back one hundred eighty days. We count past invoices, spending, recent activity, and product variety. We use the next sixty days only to check the outcome. Our models include a simple overall-rate baseline, an RFM logistic model, and a richer logistic model. All features come from before the cutoff.

## 1:00–1:30

Here are the September test results for two thousand seven hundred seventy-two customers. Average precision is zero point four nine one for the overall-rate baseline, zero point seven five zero for RFM, and zero point seven five five for the richer model. That gain is small. With two hundred seventy-eight review slots, it finds two additional repeat buyers. At other review capacities, the gain can disappear or reverse.

## 1:30–2:00

Ranking is only part of the story. We also compare predicted scores with actual purchase rates. The validation-based calibration lowers the Brier score from zero point two two two to zero point two zero nine, but the model still underestimates repeat purchases in every displayed group. This means an individual score should not be read as a precise personal probability. The chart makes that limitation visible.

## 2:00–2:30

Now I select customer fourteen thousand three hundred ninety-seven from ten stored examples. The app shows the historical model score and the customer's purchases before the cutoff. I can open the original invoice rows and see the source row identifiers. These records support the prediction explanation. The public app shows stored historical cases; it does not score arbitrary new customers live.

## 2:30–3:00

Finally, a person can inspect the evidence, accept it for analysis, or mark it for investigation. The later purchase outcome is shown only for evaluation; it was never used as a model input. This project shows an evidence-backed ranking workflow. It does not prove that contacting a customer changes behavior. For stronger evidence, we need a new, untouched time period or another retailer.
