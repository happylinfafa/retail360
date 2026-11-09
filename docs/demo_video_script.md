# Retail360: three-minute demo recording plan

Record the actual [Streamlit app](https://retail360-challenge2-nlavk4zsab8qmxgkvbfmwy.streamlit.app/Repurchase) at readable zoom. Keep the mouse movements slow. This is a recording guide, not a finished video. The public app displays ten stored real customer cases; it does not score an arbitrary new customer live.

| Time | Screen action | Simple narration |
|---|---|---|
| 0:00–0:30 | Show the poster title, then the app's Repurchase page. | “This is Retail360. Retail teams cannot review every customer. We ask whether past purchases can help identify customers who will buy again within 60 days.” |
| 0:30–1:00 | Show the 180-day history and 60-day label explanation, then the model chart. | “We use only orders before the prediction date. We compare a simple RFM model with a model that also uses recent activity and product variety. We choose the model using validation data.” |
| 1:00–1:45 | Show the model chart, fixed-capacity table, and capacity-sensitivity table. | “On the September test, average precision is 0.755 for the full model, compared with 0.750 for RFM. If an analyst reviews 278 customers, the full model finds only two more actual repeat buyers. At a 20 percent review capacity, it finds one fewer. So the practical gain is small and depends on the review budget.” |
| 1:45–2:20 | Show the calibration chart and select customer 14397. Scroll to the invoice evidence. | “The app also checks whether scores match real purchase rates. It still underestimates the observed rates. Here is a real customer. The app shows historical features and original invoice rows, so a person can inspect the evidence.” |
| 2:20–2:40 | Show the customer outcome and human-review controls. | “The future purchase is shown only as a test outcome. It was never used to make the prediction. The user can review the record before making a decision.” |
| 2:40–3:00 | Return to the poster's central finding or the app summary. | “Retail360 provides evidence-backed ranking, but we do not claim exact personal probabilities or a marketing effect. We need a new, untouched period or retailer for stronger confirmation.” |

Before submission, check that the exported video is about three minutes, that the text and voice are clear, and that it shows live interaction with the deployed app. Upload the video to an accessible location and add its URL to the README and course submission.
