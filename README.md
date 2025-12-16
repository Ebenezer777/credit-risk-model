# Credit Risk Probability Model for Alternative Data
## Credit Scoring Business Understanding
### Basel II and Model Interpretability

The Basel II Capital Accord emphasizes risk-sensitive capital requirements, requiring financial institutions to quantify, document, and justify their credit risk measurement methodologies. This regulatory focus makes model interpretability and transparency critical. An interpretable and well-documented model allows the bank to explain risk decisions to regulators, auditors, and internal stakeholders, ensures compliance, and reduces model governance risk. Black-box models without clear explanations may achieve higher performance but introduce regulatory and operational risks.

### Proxy Target Variable Justification

The dataset does not contain a direct indicator of loan default. As a result, creating a proxy target variable is necessary to approximate customer credit risk. In this project, customer behavioral patterns—such as transaction recency, frequency, and monetary value—are used to infer disengagement, which is assumed to correlate with higher default risk. However, this approach introduces business risks, including misclassification of customers and potential bias, since disengagement does not always imply inability or unwillingness to repay. Therefore, predictions based on this proxy should be treated as probabilistic risk signals rather than definitive default outcomes.

### Model Complexity Trade-offs in a Regulated Environment

Simple and interpretable models such as Logistic Regression with Weight of Evidence (WoE) provide transparency, stability, and regulatory acceptance, making them suitable for credit scoring scorecards. However, they may fail to capture complex non-linear relationships in alternative data. More complex models like Gradient Boosting can deliver higher predictive performance but reduce explainability and increase governance complexity. In regulated financial contexts, the trade-off involves balancing predictive accuracy with explainability, auditability, and regulatory compliance.
