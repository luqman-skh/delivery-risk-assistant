# Implementation Plan: SLM-Powered Delivery Risk Assistant

**Project:** SLM-powered Delivery Risk Assistant for HapperTrade Commerce GmbH  
**Document Version:** 1.0.0  
**Status:** Approved for Execution  

---

## 1. Project Overview & Business Scenario

### 1.1 Context & Problem Statement
HapperTrade Commerce GmbH is an international retailer handling high transaction volumes across diverse product lines, global shipping corridors, and fulfillment modes. Currently, customer support and order-fulfillment staff monitor orders manually, pulling records across fragmented systems to detect delays and decide on mitigation actions.

This manual process leads to:
* Delayed identification of delivery risks.
* Inconsistent recovery decisions across customer service agents.
* Unnecessary manual interventions on low-risk orders.
* High operational costs and poor customer satisfaction.

### 1.2 System Objectives
The objective is to implement a robust, end-to-end **LLMOps-driven Multi-Agent Delivery Risk Assistant** that:
1. Identifies customer orders at elevated risk of late delivery before shipment failure.
2. Retrieves operational order, shipping, customer, and product information dynamically using tool calling.
3. Evaluates delivery risk and operational context against defined **Order Recovery Policies**.
4. Recommends standardized recovery actions with human-readable rationale and transparency.
5. Enables Human-in-the-Loop (HITL) review for high-risk or ambiguous cases.
6. Evaluates and compares a **General-Purpose SLM** against a **Specialized Small Decision Model (Clef-Flash / Tabular SLM)** across model quality, policy compliance, inference latency, and operational cost.
7. Supports policy drift/lifecycle updates with automated validation and deployment rollback gating.

---

## 2. Requirements Matrix & Traceability

### 2.1 Functional Requirements (FR)
| ID | Requirement Description | Implementation Mechanism |
| :--- | :--- | :--- |
| **FR1** | Identify orders with elevated delivery-delay risk. | Tabular ML/SLM Classifier (XGBoost / LightGBM) trained on leak-free order features. |
| **FR2** | Provide relevant operational information for identified orders. | Operational context builder aggregating customer, order, items, and logistics metrics. |
| **FR3** | Retrieve additional order, shipping, and historical data when required. | Tool-calling framework (Order Inquiry, Shipping History, Customer Profile, Policy Lookup). |
| **FR4** | Recommend one of the predefined response actions. | Structured decision generation via General-Purpose SLM & Small Decision Model. |
| **FR5** | Provide supporting information and reasoning for proposed actions. | Chain-of-thought / evidence citation in agent output with risk breakdown. |
| **FR6** | Enable human users to accept, reject, or override proposed actions. | Human-in-the-Loop (HITL) feedback schema and review queue API. |

### 2.2 Non-Functional Requirements (NFR)
| ID | Requirement | Target Threshold | Validation Strategy |
| :--- | :--- | :--- | :--- |
| **NFR1** | Performance & Latency | Risk prediction < 100ms; Total end-to-end decision < 3.5s per order. | Latency profiling per pipeline stage under batch and single-item requests. |
| **NFR2** | Reliability & Robustness | Graceful handling of tool/agent timeouts; tool failure rate < 1.0%. | Fault injection tests, schema validation, fallback to Human Escalation. |
| **NFR3** | Transparency & Traceability | 100% of decisions include input features, policy citation, and reasoning. | Audit logging of prompt, tool calls, retrieved payloads, and outputs. |
| **NFR4** | Decision Quality & Compliance | Policy compliance ≥ 92%; Delay recall ≥ 85%; Escalation recall ≥ 90%. | Automated benchmark test suite over synthetic & historical test splits. |

---

## 3. Metrics & KPI Framework

| Layer | KPI Name | Formula / Definition | Target Baseline |
| :--- | :--- | :--- | :--- |
| **Business** | Early Delayed Identification % | $\frac{\text{Delayed orders flagged before scheduled date}}{\text{Total actual delayed orders}} \times 100$ | $\ge 85\%$ |
| | Reduction in Manual Monitoring | Reduction in hours spent on non-problematic orders | $\ge 40\%$ |
| | High-Risk Recovery Accuracy | % high-risk orders assigned optimal recovery mitigation | $\ge 90\%$ |
| | Unnecessary Interventions Reduction | % reduction in false-alarm interventions on on-time orders | $\ge 35\%$ |
| **Application & Model** | Delay-Risk Recall & F1 | Recall and harmonic mean for binary `Late_delivery_risk` | Recall $\ge 0.85$, F1 $\ge 0.80$ |
| | Recovery-Action Accuracy | $\frac{\text{Decisions matching ground-truth policy}}{\text{Total cases}}$ | $\ge 90\%$ |
| | Macro-F1 (Recovery Actions) | Balanced harmonic mean across all action categories | $\ge 0.85$ |
| | Policy-Compliance Rate | Actions strictly adhering to active operational policy rules | $\ge 95\%$ |
| | Tool-Selection Accuracy | % of appropriate tools invoked for operational questions | $\ge 95\%$ |
| | Escalation Recall | % of conflicting/uncertain cases correctly escalated | $\ge 92\%$ |
| **Operational & Cost** | Risk Prediction Latency | Model inference duration ($P_{50}, P_{95}$) | $P_{95} \le 120\text{ms}$ |
| | Recovery Decision Latency | SLM decision generation duration ($P_{50}, P_{95}$) | $P_{95} \le 2.5\text{s}$ |
| | End-to-End Processing Time | Tool retrieval + risk scoring + SLM reasoning | $P_{95} \le 3.5\text{s}$ |
| | Tool-Call Failure Rate | $\frac{\text{Failed or malformed tool invocations}}{\text{Total tool invocations}}$ | $< 0.5\%$ |
| | Token & Resource Consumption | Average prompt + completion tokens / memory footprint | General SLM vs Clef-Flash |

---

## 4. Data Architecture & Leakage Prevention Strategy

### 4.1 Data Sources
* **Data File:** `data/DataCoSupplyChainDataset.csv` (180,519 records, 53 attributes)
* **Metadata:** `data/DescriptionDataCoSupplyChain.csv`
* **Access Logs:** `data/tokenized_access_logs.csv`

### 4.2 Data Leakage Strict Firewall
The dataset contains attributes recorded *after* delivery is completed. Including them in predictive modeling causes severe target leakage.
* **TARGET ATTRIBUTE:** `Late_delivery_risk` (0 = On-Time / Early, 1 = Late).
* **FORBIDDEN (Leakage) Features:**
  * `Delivery Status` (reveals outcome: "Advance shipping", "Late delivery", "Shipping canceled", "Shipping on time")
  * `Days for shipping (real)` (reveals actual transit time compared to scheduled)
  * `Shipping date (DateOrders)` (often identical to delivery or derived post-transit)
* **PERMITTED PREDICTION FEATURES (Available at or before order dispatch):**
  * **Order Timing & Scheduling:** `Days for shipment (scheduled)`, order hour, order day of week, order month.
  * **Order & Product Characteristics:** `Product Price`, `Product Category Id`, `Category Name`, `Order Item Quantity`, `Order Item Total`, `Sales per customer`, `Order Profit Per Order`.
  * **Shipping Configuration:** `Shipping Mode` (Standard Class, First Class, Second Class, Same Day), `Type` (DEBIT, TRANSFER, PAYMENT, CASH).
  * **Geographical Corridors:** `Customer Country`, `Customer State`, `Customer Segment`, `Order Country`, `Order Region`, `Order State`, `Market`.
  * **Customer History:** Prior orders count, historical delay rate per customer/market corridor.

---

## 5. Phase-by-Phase Implementation Roadmap

```
+---------------------------------------------------------------------------------------------------+
|                                     THREE-PHASE LIFECYCLE ROADMAP                                 |
+---------------------------------------------------------------------------------------------------+
| PHASE 1: Baseline Architecture                                                                    |
|  * Clean dataset & enforce strict leakage firewall                                                |
|  * Train & evaluate tabular risk model (XGBoost/LightGBM)                                         |
|  * Implement Operational Tools (Customer, Order, History, Policy)                                 |
|  * Multi-Agent Workflow v1 with General-Purpose SLM                                               |
|  * Deploy Baseline v1 & establish KPI benchmarks                                                  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 2: Decision Model Alternative & Policy Drift                                                |
|  * Integrate Small Decision Model (Clef-Flash / Compact Decision SLM)                             |
|  * Benchmark General-Purpose SLM vs. Small Decision Model on Policy v1                            |
|  * Implement Policy v2 (Customer Tiering, Expedited Rerouting, Critical Cost Thresholds)          |
|  * Extend operational tools (Inventory & Courier SLA inquiry)                                     |
|  * Adapt multi-agent workflow for Policy v2                                                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 3: Adaptation Evaluation, Gated Deployment & Report                                         |
|  * Comparative evaluation under Policy v2 (Compliance, Macro-F1, Regression on Legacy cases)      |
|  * Trade-off analysis: Decision Quality vs Latency vs Cost/Resource Usage                         |
|  * Production Decision Gate & Automated Rollback mechanism                                        |
|  * Generate UML Diagrams (.xml format): Class, Component, Sequence diagrams                       |
|  * Compile Comprehensive Final Report & KPI verification                                          |
+---------------------------------------------------------------------------------------------------+
```

---

### Phase 1: Foundation, Risk ML Model & Multi-Agent Baseline (General-Purpose SLM)

#### Objectives
1. Prepare leak-free dataset with train/validation/test splits.
2. Build, optimize, and serialize a tabular delivery risk prediction model.
3. Build mockable operational tools representing enterprise microservices.
4. Formalize **Order Recovery Policy v1**.
5. Implement the Multi-Agent Coordination Workflow using a General-Purpose SLM.
6. Evaluate and deploy Pipeline v1; log baseline KPIs.

#### Work Packages & Implementation Steps

* **WP 1.1: Data Engineering & Feature Store Pipeline**
  * Script: `src/data/preprocess.py`
  * Action: Ingest `DataCoSupplyChainDataset.csv`, drop leakage columns (`Days for shipping (real)`, `Delivery Status`), handle categorical encoding, compute corridor risk aggregates, and partition into Train (70%), Validation (15%), Test (15%) stratified by `Late_delivery_risk`.
  * Artifacts: `data/processed/train.parquet`, `data/processed/test.parquet`, `data/processed/feature_metadata.json`.

* **WP 1.2: Delivery Risk Tabular Classifier**
  * Script: `src/models/risk_classifier.py`
  * Action: Train XGBoost classifier on engineered features. Tune probability threshold to prioritize **Delay-risk recall** ($\ge 0.85$) while keeping false alarms low.
  * Output: Model artifact `models/risk_xgb_v1.json`, evaluation report with ROC-AUC, Recall, Precision, and F1.

* **WP 1.3: Operational Data Tools Suite**
  * Module: `src/tools/operational_tools.py`
  * Implement tools with standardized JSON schemas:
    1. `get_order_details(order_id)`: Fetches items, amounts, destination, shipping mode.
    2. `get_customer_profile(customer_id)`: Fetches customer segment, historical orders, lifetime value.
    3. `get_shipping_corridor_history(origin, destination, shipping_mode)`: Historical delay rates for this specific route.
    4. `get_recovery_policy(policy_version)`: Returns active policy rules.

* **WP 1.4: Order Recovery Policy v1 Definition**
  * Specification file: `configs/policies/policy_v1.yaml`
  * Action Space:
    1. `CONTINUE_MONITORING`: Risk score < 0.40.
    2. `CONTACT_LOGISTICS`: Risk score $\ge$ 0.40, standard order value (< $200).
    3. `PRIORITIZE_SHIPMENT`: Risk score $\ge$ 0.40, high-value order ($\ge$ $200).
    4. `NOTIFY_CUSTOMER`: High risk with severe customer impact (e.g. repeat delayed customer or VIP segment).
    5. `HUMAN_ESCALATION`: Conflicting information, tool query failures, or borderline cases (0.45 - 0.55 uncertainty band).

* **WP 1.5: Multi-Agent Workflow v1 (General-Purpose SLM)**
  * Architecture: `src/agents/coordinator.py`, `src/agents/risk_agent.py`, `src/agents/decision_agent.py`.
  * Coordinator dispatches risk assessment, calls operational tools, prompts General-Purpose SLM with consolidated context, and outputs a structured Pydantic response:
    ```python
    class DeliveryRiskAssessment(BaseModel):
        order_id: int
        risk_score: float
        risk_level: Literal["LOW", "MEDIUM", "HIGH"]
        recommended_action: Literal["CONTINUE_MONITORING", "CONTACT_LOGISTICS", "PRIORITIZE_SHIPMENT", "NOTIFY_CUSTOMER", "HUMAN_ESCALATION"]
        supporting_evidence: list[str]
        policy_citation: str
        confidence: float
        escalate_to_human: bool
    ```

* **WP 1.6: Phase 1 Evaluation & Deployment Baseline**
  * Script: `src/eval/evaluate_phase1.py`
  * Metrics computed on test dataset (1,000 holdout cases).
  * Generate `reports/phase1_baseline_kpi.json` and deploy v1 pipeline endpoint.

#### Phase 1 Verification Checklist
- [ ] Data preprocessing asserts no presence of `Delivery Status` or `Days for shipping (real)` in train features.
- [ ] Tabular risk classifier achieves Delay Recall $\ge 0.85$ on test set.
- [ ] Operational tools return valid schemas and handle missing order IDs gracefully.
- [ ] Multi-Agent workflow runs end-to-end for sample orders and returns valid JSON matching `DeliveryRiskAssessment`.
- [ ] Phase 1 baseline KPIs documented and committed.

---

### Phase 2: Small Decision Model Integration & Policy Evolution

#### Objectives
1. Introduce a specialized **Small Decision Model** (Clef-Flash / Compact Decision SLM) as an alternative response-action selector.
2. Provide identical input signals (risk score, operational context, policy rules) to both general SLM and decision model.
3. Benchmark General-Purpose SLM vs Small Decision Model on Policy v1.
4. Evolve to **Order Recovery Policy v2** (introducing business changes: customer loyalty tiers, expedited rerouting, cost thresholds).
5. Extend operational tools and agentic workflow to reflect Policy v2.

#### Work Packages & Implementation Steps

* **WP 2.1: Small Decision Model Adapter & Prompt Optimization**
  * Module: `src/models/decision_model.py`
  * Implement inference adapter supporting Clef-Flash / compact local SLM / structured classifier.
  * Standardize zero-shot and few-shot system prompts with strict JSON schema constraints.
  * Ensure identical inputs between General-Purpose SLM and Small Decision Model for exact apples-to-apples comparison.

* **WP 2.2: Side-by-Side Comparative Benchmarking (Policy v1)**
  * Script: `src/eval/benchmark_models.py`
  * Test both models across 1,000 test cases on Policy v1.
  * Record: Action accuracy, Macro-F1, Policy compliance, Escalation recall, Latency ($P_{50}, P_{95}$), and token consumption.
  * Output: `reports/phase2_model_comparison_v1.json`.

* **WP 2.3: Order Recovery Policy v2 Specification (Policy Drift)**
  * Specification file: `configs/policies/policy_v2.yaml`
  * New Policy Rules (reflecting realistic business requirements change):
    1. `CONTINUE_MONITORING`: Risk score < 0.35 (tightened threshold).
    2. `EXPEDITE_CARRIER`: High risk ($\ge 0.35$), high value ($\ge \$300$), or Platinum/Corporate customer segment.
    3. `RE_ROUTE_WAREHOUSE`: High risk ($\ge 0.35$), regional warehouse stock available within 24h.
    4. `PROACTIVE_CUSTOMER_CREDIT`: Severe customer impact with > 2 previous delayed orders.
    5. `CONTACT_LOGISTICS`: High risk ($\ge 0.35$), standard value order.
    6. `HUMAN_ESCALATION`: Data conflict, stock unknown, or risk score in uncertainty band (0.45 - 0.52).

* **WP 2.4: Tool Suite Expansion for Policy v2**
  * Script: `src/tools/extended_tools.py`
  * Additional tools:
    * `check_regional_inventory(product_id, customer_region)`: Checks alternate fulfillment nodes.
    * `get_customer_loyalty_tier(customer_id)`: Returns Corporate, Platinum, Gold, or Standard tier.
    * `get_courier_sla(shipping_mode, corridor)`: Returns SLA guarantees and cutoff times.

* **WP 2.5: Multi-Agent Workflow Update**
  * Update coordinator to dynamically route operational inquiries based on the active policy version.
  * Modularize prompt templates: `configs/prompts/decision_v1.jinja2` and `decision_v2.jinja2`.

#### Phase 2 Verification Checklist
- [ ] Small Decision Model generates valid actions conforming to the schema.
- [ ] Comparative benchmark between General SLM and Decision Model completed on identical test splits.
- [ ] Policy v2 formalized and loaded cleanly by the policy engine.
- [ ] Extended tools (`check_regional_inventory`, `get_customer_loyalty_tier`) tested and functional.
- [ ] Agents successfully invoke new tools and incorporate extra context into decision payloads.

---

### Phase 3: Policy Adaptation Benchmark, Gated Deployment & Report

#### Objectives
1. Evaluate whether both models adapt to Policy v2 while maintaining stability on legacy patterns.
2. Conduct full quality-cost-latency trade-off analysis.
3. Implement an **Automated Deployment Decision Gate** with instant rollback capability.
4. Generate UML architectural blueprints (.xml format for Class, Component, and Sequence diagrams conforming to UML standard).
5. Compile comprehensive project report and code documentation.

#### Work Packages & Implementation Steps

* **WP 3.1: Policy v2 Adaptation & Regression Evaluation**
  * Script: `src/eval/evaluate_phase3_adaptation.py`
  * Evaluate both General-Purpose SLM and Small Decision Model against:
    * Policy Compliance Rate on new Policy v2 test set (300 cases).
    * Regression / Stability on legacy orders that did not change policy.
    * Out-of-distribution & adversarial edge cases (missing attributes, contradictory signals).
  * Measure Escalation Recall: ensuring ambiguous cases are safely sent to human operators.

* **WP 3.2: Quality-Cost-Latency Trade-Off Analysis**
  * Build trade-off matrix:
    * Decision Quality: Policy compliance %, Macro-F1, Escalation recall.
    * Latency: $P_{50}, P_{95}, P_{99}$ latency in milliseconds.
    * Resource Efficiency: Token usage per decision, memory footprint, compute cost per 10k orders.
  * Determine the optimal production candidate based on pre-established criteria:
    $$\text{Deploy Decision Model if: } \text{Compliance}_{\text{DM}} \ge 0.92 \times \text{Compliance}_{\text{SLM}} \text{ AND Latency}_{\text{DM}} \le 0.5 \times \text{Latency}_{\text{SLM}}$$

* **WP 3.3: Deployment Decision Gate & Rollback Mechanism**
  * Module: `src/deployment/gatekeeper.py`
  * Automated canary & gating validation:
    * Runs test suite before model switch.
    * If candidate fails any safety or compliance threshold, the gatekeeper automatically rolls back to the previously validated model version.
    * Emits structured deployment audit log: `logs/deployment_audit.json`.

* **WP 3.4: Architectural Blueprints Generation (.xml standard UML)**
  * Generate standard XML UML files in `docs/diagrams/`:
    1. `uml_class_diagram.xml`: Information entities, metadata, artifacts, relations (Order, RiskModel, OperationalTool, Policy, RecoveryDecision, EvaluationRun).
    2. `uml_component_diagram.xml`: Technical components, interfaces, dependencies (RiskClassifier, ToolRegistry, AgentCoordinator, ModelAdapter, PolicyEngine, Gatekeeper).
    3. `uml_sequence_diagram.xml`: Runtime interaction and message flows for order risk evaluation, tool calls, decision routing, and human escalation/override.

* **WP 3.5: Final Extensive Report & Documentation**
  * Artifact: `reports/FINAL_PROJECT_REPORT.md` (and exportable doc) covering:
    1. Extended Functional & Non-Functional Requirements.
    2. LLMOps lifecycle stages and UML activity diagrams.
    3. Architecture blueprints and data lineage.
    4. Business scenario outputs, case studies, and recommendations.
    5. Complete KPI scorecard across all 3 phases.

#### Phase 3 Verification Checklist
- [ ] Evaluation suite completes for Policy v2 on both models with macro-F1 and compliance scores.
- [ ] Deployment Gatekeeper validates candidate model against predefined thresholds and demonstrates rollback on induced failure.
- [ ] UML Class, Component, and Sequence diagrams generated in valid standard `.xml` format.
- [ ] Final comprehensive report completed with full KPI tables and business recommendations.

---

## 6. Software Architecture & Directory Layout

```
LLMOps-Delivery-Risk-Assistant/
├── configs/
│   ├── model_config.yaml           # Hyperparameters for XGBoost & SLMs
│   ├── policies/
│   │   ├── policy_v1.yaml          # Order Recovery Policy v1
│   │   └── policy_v2.yaml          # Order Recovery Policy v2 (drifted)
│   └── prompts/
│       ├── coordinator_v1.jinja2   # Agent prompt for v1
│       └── coordinator_v2.jinja2   # Agent prompt for v2
├── data/
│   ├── DataCoSupplyChainDataset.csv
│   ├── DescriptionDataCoSupplyChain.csv
│   ├── tokenized_access_logs.csv
│   └── processed/                  # Leak-free train, val, test parquets
├── docs/
│   └── diagrams/                   # UML Architectural Blueprints (.xml)
│       ├── uml_class_diagram.xml
│       ├── uml_component_diagram.xml
│       └── uml_sequence_diagram.xml
├── models/
│   ├── risk_xgb_v1.json            # Serialized XGBoost model
│   └── artifacts/                  # Scalers, encoders, and threshold configs
├── reports/
│   ├── phase1_baseline_kpi.json
│   ├── phase2_model_comparison.json
│   ├── phase3_tradeoff_analysis.json
│   └── FINAL_PROJECT_REPORT.md     # Complete deliverable report
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coordinator.py          # Multi-agent orchestrator
│   │   ├── risk_agent.py           # Tabular risk assessment agent
│   │   └── decision_agent.py       # Recovery response action agent
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Data cleaning & leakage firewall
│   │   └── feature_pipeline.py     # Feature engineering & scaling
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── gatekeeper.py           # Gated deployment & automated rollback
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Business, model, and operational KPIs
│   │   ├── benchmark.py            # Comparative evaluation runner
│   │   └── synthetic_cases.py      # Edge case & policy validation generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── risk_classifier.py      # Tabular risk predictor wrapper
│   │   └── decision_model.py       # Small decision model & general SLM clients
│   └── tools/
│       ├── __init__.py
│       ├── operational_tools.py    # Order, customer, corridor tools
│       └── extended_tools.py       # Inventory, loyalty, courier SLA tools
├── tests/
│   ├── test_data_leakage.py        # Strict assertions against target leakage
│   ├── test_risk_model.py          # Unit tests for risk classifier
│   ├── test_tools.py               # Tool schema & failure handling tests
│   ├── test_policies.py            # Policy compliance unit tests
│   └── test_pipeline_e2e.py        # End-to-end integration test
├── IMPLEMENTATION_PLAN.md          # This implementation plan
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project overview and run guide
```

---

## 7. Risk Analysis & Mitigation Strategies

| Risk Factor | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Data Leakage in Delivery Risk** | Critical | High | Hardcoded column drop list with automated unit tests asserting absence of leakage features (`Delivery Status`, real transit days) before any training. |
| **SLM Non-Deterministic Output** | High | Medium | Strict Pydantic schema validation; temperature = 0.0; automated fallback to `HUMAN_ESCALATION` if JSON parsing fails. |
| **Policy Drift Non-Compliance** | High | Medium | Automated regression test suite on policy changes; dynamic few-shot examples illustrating edge cases for the active policy. |
| **High Inference Latency of SLMs** | Medium | Medium | Benchmark Small Decision Model (Clef-Flash / compact local model) to cut latency by >50%; implement response caching for identical corridor risks. |
| **Computational Resource Constraints** | Medium | Low | Use quantized/compact SLMs and fast gradient-boosted trees (XGBoost) operable locally without dedicated GPU clusters. |

---

## 8. Immediate Next Steps: How to Start Development

With this plan established, development proceeds in clear, testable stages:

1. **Phase 1 Step 1:** Set up virtual environment and install dependencies (`scikit-learn`, `xgboost`, `pandas`, `pydantic`, `jinja2`).
2. **Phase 1 Step 2:** Execute data leakage audit and preprocessing script on `data/DataCoSupplyChainDataset.csv`.
3. **Phase 1 Step 3:** Train and calibrate the XGBoost delivery risk model and verify Recall $\ge 0.85$.
4. **Phase 1 Step 4:** Implement operational tools and the Multi-Agent Coordinator v1.
5. **Phase 1 Step 5:** Run baseline evaluation and verify all Phase 1 KPIs.
