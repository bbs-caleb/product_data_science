#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import textwrap

REPO_NAME = "product_data_science"

@dataclass(frozen=True)
class Task:
    slug: str
    title: str

@dataclass(frozen=True)
class Domain:
    folder: str
    name: str
    readme: str
    tasks: list[Task]

def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def md_domain_readme(summary: str, scope: list[str], skills: list[str], deliverables: list[str], dod: list[str]) -> str:
    def bullets(xs: list[str]) -> str:
        return "\n".join([f"- {x}" for x in xs])
    return textwrap.dedent(f"""
    Summary
    {summary}

    Scope
    {bullets(scope)}

    Key Skills
    {bullets(skills)}

    Deliverables
    {bullets(deliverables)}

    Definition of Done
    {bullets(dod)}
    """).strip()

def md_task_readme(domain_name: str, task_title: str) -> str:
    return textwrap.dedent(f"""
    Summary
    {task_title}

    Domain
    {domain_name}

    Problem Statement
    TODO: Describe the business context and the exact task requirements.

    Inputs and Constraints
    TODO: Data schema, granularity, constraints, edge cases.

    Approach
    TODO: High-level solution outline. Link to TECH.md for details.

    Validation
    TODO: Checks, tests, sanity checks, evaluation metrics.

    Result
    TODO: Key outputs and brief interpretation.

    Runbook
    - TODO: How to reproduce (commands, environment, data assumptions).

    Artifacts
    - reports/: figures, tables, final outputs
    - sql/: queries (if applicable)
    - notebooks/: exploration (if applicable)
    - src/: production-style code (if applicable)
    - tests/: automated tests (if applicable)
    """).strip()

def md_task_tech(task_title: str) -> str:
    return textwrap.dedent(f"""
    Technical Notes
    {task_title}

    Assumptions
    TODO

    Data Contract
    TODO: Grain, primary keys, filters, missingness strategy.

    Methodology
    TODO: Algorithms, formulas, statistical tests, model training details.

    Edge Cases
    TODO

    Complexity
    TODO: Runtime/memory considerations where relevant.

    Risks and Limitations
    TODO

    References
    TODO
    """).strip()

def root_readme(domains: list[Domain]) -> str:
    domain_lines = "\n".join([f"- `{d.folder}/` — {d.name}" for d in domains])
    return textwrap.dedent(f"""
    {REPO_NAME}

    Summary
    Product Data Science monorepo. Organized by product/business domains with simulator tasks as reproducible case studies.

    Structure
    {domain_lines}
    - `catalog.yml` — task registry (domain, status, tags)
    - `PORTFOLIO.md` — curated highlights for interviews

    How to Use
    - Each domain contains task folders directly (no extra nesting):
      `01_product_metrics/wau/`, `08_experimentations/quantile_t_test/`, etc.
    - Each task includes `README.md` (case card) and `TECH.md` (implementation details)
    - Repro instructions live in the task "Runbook" section
    """).strip()

def portfolio_md() -> str:
    return textwrap.dedent("""
    Portfolio Highlights

    Summary
    Curated list of the most interview-relevant case studies. Update this file as you complete tasks.

    Recommended Highlights (fill with links)
    - Uplift / targeting: 04_marketing_and_growth/uplift_tree
    - Churn modeling: 03_customer_value_management/churn_rate_i_sql + churn_rate_ii_bootstrap
    - MMM / budget allocation: 04_marketing_and_growth/marketing_mix_modeling_mmm
    - Pricing optimization: 07_pricing_revenue_optimization/target_margin, coca_cola_principle
    - Forecasting with uncertainty: 06_demand_supply_forecasting/boosting_uncertainty
    - Experiment design & inference: 08_experimentations/smart_link_ii_ab_design
    - Recsys / embeddings / ANN: 05_customer_segmentation_scoring/sku_embeddings, approximate_knn

    Format for each highlight
    - Context
    - What you built
    - Validation / metric
    - Result
    - Link to task folder
    """).strip()

def gitignore() -> str:
    return textwrap.dedent("""
    # Python
    __pycache__/
    *.pyc
    *.pyo
    *.pyd
    .Python
    .venv/
    venv/
    env/
    .mypy_cache/
    .pytest_cache/
    .ruff_cache/

    # Notebooks
    .ipynb_checkpoints/

    # OS
    .DS_Store
    Thumbs.db

    # Outputs / data
    outputs/
    data/
    *.parquet
    *.csv
    *.sqlite
    *.db

    # Secrets
    .env
    **/*secret*
    """).strip()

def catalog_yaml(domains: list[Domain]) -> str:
    lines = []
    lines.append(f"generated_at: '{datetime.utcnow().isoformat()}Z'")
    lines.append("tasks:")
    for d in domains:
        for t in d.tasks:
            lines.append(f"  - domain: '{d.folder}'")
            lines.append(f"    slug: '{t.slug}'")
            lines.append(f"    title: '{t.title}'")
            lines.append("    status: 'planned'")
            lines.append("    tags: []")
    return "\n".join(lines)

def make_task_folder(domain_root: Path, domain_name: str, task: Task) -> None:
    task_root = domain_root / task.slug
    ensure_dir(task_root)
    for sub in ["src", "notebooks", "sql", "tests", "reports"]:
        ensure_dir(task_root / sub)
    write_file(task_root / "README.md", md_task_readme(domain_name, task.title))
    write_file(task_root / "TECH.md", md_task_tech(task.title))

def main() -> None:
    root = Path(".").resolve()
    repo_root = root / REPO_NAME if root.name != REPO_NAME else root
    ensure_dir(repo_root)

    domains: list[Domain] = [
        Domain(
            folder="01_product_metrics",
            name="Product Metrics",
            readme=md_domain_readme(
                summary="Product health metrics and behavioral analytics used to monitor usage, engagement, and experience quality.",
                scope=["Activity and engagement metrics", "Session analytics and product signals", "Reporting-ready outputs and dashboards"],
                skills=["SQL analytics and metric design", "Data validation checks", "Documentation of metric contracts"],
                deliverables=["Metric definitions and calculation queries", "Reproducible scripts/notebooks", "Artifacts under reports/"],
                dod=["Metric contract documented (definition, grain, filters)", "Reproducible runbook included", "Sanity checks implemented where applicable"],
            ),
            tasks=[
                Task("wau", "WAU"),
                Task("dau", "DAU"),
                Task("proxy_metrics", "Proxy Metrics"),
                Task("session_analysis", "Session Analysis"),
                Task("net_promoter_score", "Net Promoter Score"),
                Task("promo_calendar", "Promo Calendar"),
                Task("time_series_dashboard", "Time-Series Dashboard"),
            ],
        ),
        Domain(
            folder="02_unit_economics",
            name="Unit Economics",
            readme=md_domain_readme(
                summary="Revenue and subscription analytics focusing on core unit economics metrics.",
                scope=["ARPU/AOV and transaction metrics", "Subscription revenue and MRR", "Revenue dashboards and reconciliation mindset"],
                skills=["SQL for finance metrics", "Time aggregation and window logic", "Validation and reconciliation"],
                deliverables=["Metric computation SQL + validation notes", "Dashboard-ready outputs", "Short business interpretation per task"],
                dod=["Assumptions documented", "Time period handling validated", "Outputs match expected formats and examples"],
            ),
            tasks=[
                Task("arpu_aov", "ARPU & AOV"),
                Task("average_check", "Average Check"),
                Task("monthly_recurring_revenue_mrr", "Monthly Recurring Revenue (MRR)"),
                Task("payments_dashboard", "Payments Dashboard"),
            ],
        ),
        Domain(
            folder="03_customer_value_management",
            name="Customer Value Management (Lifecycle)",
            readme=md_domain_readme(
                summary="Lifecycle analytics for retention and churn to support customer value management (CVM).",
                scope=["Retention and cohort retention", "Churn dataset engineering and modeling validation", "Lifecycle definitions and segmentation support"],
                skills=["Cohort design and churn labeling", "Bootstrap and evaluation", "Leakage checks and stable definitions"],
                deliverables=["Cohort/retention queries", "Churn dataset build and modeling artifacts", "Validation reports under reports/"],
                dod=["Correct cohort/grain design", "Leakage checks documented", "Clear definitions of retention/churn and evaluation"],
            ),
            tasks=[
                Task("retention_rate", "Retention Rate"),
                Task("cohort_retention", "Cohort Retention"),
                Task("churn_rate_i_sql", "Churn Rate I: SQL"),
                Task("churn_rate_ii_bootstrap", "Churn Rate II: Bootstrap"),
            ],
        ),
        Domain(
            folder="04_marketing_and_growth",
            name="Marketing and Growth",
            readme=md_domain_readme(
                summary="Marketing efficiency and growth analytics including uplift and budget optimization.",
                scope=["Uplift modeling and targeting", "Marketing mix modeling and budget allocation", "Bandits and marketing monitoring"],
                skills=["Causal framing and uplift evaluation", "Optimization under business constraints", "Operational monitoring and alerting logic"],
                deliverables=["Models/notebooks with evaluation", "Budget allocation and targeting outputs", "Monitoring rules and reports where applicable"],
                dod=["Objective and success metrics documented", "Reproducible code and results", "Clear interpretation and actionable outputs"],
            ),
            tasks=[
                Task("uplift_tree", "Uplift Tree"),
                Task("marketing_mix_modeling_mmm", "Marketing Mix Modeling (MMM)"),
                Task("smart_link_i_epsilon_greedy", "Smart-Link I: Epsilon-Greedy"),
                Task("dead_ad_alert", "Dead Ad Alert"),
                Task("asymmetric_metrics", "Asymmetric Metrics"),
                Task("gmroi_optimization", "GMROI Optimization"),
            ],
        ),
        Domain(
            folder="05_customer_segmentation_scoring",
            name="Customer Segmentation and Scoring",
            readme=md_domain_readme(
                summary="Segmentation, scoring, and personalization foundations: ranking, retrieval, embeddings, matching.",
                scope=["Segmentation and scoring models", "Ranking/retrieval metrics and evaluation", "Embeddings, ANN, and matching pipelines"],
                skills=["Unsupervised learning and scoring", "Offline evaluation for ranking/recsys", "Approximate nearest neighbors and embeddings"],
                deliverables=["Pipelines and evaluation artifacts", "Metric implementations", "Reports and reproducible code"],
                dod=["Evaluation setup is correct and documented", "Pipelines are reproducible", "Assumptions and constraints are explicit"],
            ),
            tasks=[
                Task("user_segmentation", "User Segmentation"),
                Task("cold_start", "Cold Start"),
                Task("decision_tree_scoring", "Decision Tree (Scoring)"),
                Task("k_means_posterization", "K-Means Posterization"),
                Task("match_groups", "Match Groups"),
                Task("match_items", "Match Items"),
                Task("similar_item_price", "Similar Item Price"),
                Task("recsys_live_streaming_platform", "RecSys Live-Streaming Platform"),
                Task("sku_uniqueness", "SKU Uniqueness"),
                Task("sku_embeddings", "SKU Embeddings"),
                Task("approximate_knn", "Approximate kNN"),
                Task("ndcg", "NDCG"),
                Task("recall_at_k", "Recall@K"),
                Task("bm25", "BM-25"),
                Task("smart_cart_ranking", "Smart Cart Ranking"),
            ],
        ),
        Domain(
            folder="06_demand_supply_forecasting",
            name="Demand and Supply Forecasting",
            readme=md_domain_readme(
                summary="Forecasting and supply/inventory analytics, including uncertainty and monitoring.",
                scope=["Demand forecasting and backtesting", "Supply/inventory analytics", "Uncertainty estimation and monitoring"],
                skills=["Time-series feature work and validation", "Uncertainty framing and reporting", "Monitoring and drift concepts"],
                deliverables=["Forecasting pipeline + evaluation", "Uncertainty/drift artifacts", "Clear runbooks and reports"],
                dod=["Train/validation split justified", "Backtesting results included", "Monitoring metrics defined where relevant"],
            ),
            tasks=[
                Task("demand_forecast", "Demand Forecast"),
                Task("stock_supply", "Stock Supply"),
                Task("boosting_uncertainty", "Boosting Uncertainty"),
                Task("gradient_boosting", "Gradient Boosting"),
                Task("data_drift", "Data Drift"),
                Task("temporal_fusion_transformers", "Temporal Fusion Transformers"),
                Task("stocks_postprocessing", "Stocks (Postprocessing)"),
            ],
        ),
        Domain(
            folder="07_pricing_revenue_optimization",
            name="Pricing and Revenue Optimization",
            readme=md_domain_readme(
                summary="Pricing analytics and optimization: competitor pricing, parsing, elasticity, margin constraints.",
                scope=["Dynamic pricing and constrained optimization", "Competitor pricing and parsing", "Demand-price relationship features"],
                skills=["Optimization under constraints", "Feature engineering for pricing", "Reliable parsing and aggregation"],
                deliverables=["Pricing algorithms and reports", "Parsing/aggregation logic", "Recommendation-ready outputs"],
                dod=["Constraints documented (floors/ceilings/margins)", "Input validation included", "Reproducible runbook present"],
            ),
            tasks=[
                Task("surge_pricing", "Surge Pricing"),
                Task("elasticity_feature", "Elasticity Feature"),
                Task("competitor_price", "Competitor Price"),
                Task("price_parser", "Price Parser"),
                Task("target_margin", "Target Margin"),
                Task("coca_cola_principle", "Coca-Cola Principle"),
            ],
        ),
        Domain(
            folder="08_experimentations",
            name="Experimentations",
            readme=md_domain_readme(
                summary="Experiment design and statistical measurement: A/B testing, hypothesis testing, causal analysis.",
                scope=["A/B test design and simulation", "Hypothesis testing including quantile tests", "Causal impact analysis"],
                skills=["Experiment design and inference", "Assumptions and multiple testing awareness", "Decision criteria and interpretation"],
                deliverables=["Test design artifacts and code", "Statistical test implementations", "Decision criteria documented in README/TECH"],
                dod=["Hypotheses and metrics stated upfront", "Correct procedure and assumptions", "Reproducible outputs"],
            ),
            tasks=[
                Task("smart_link_ii_ab_design", "Smart-Link II: A/B Design"),
                Task("kaggle_ab_test", "Kaggle: A/B Test"),
                Task("t_tested_features", "T-Tested Features"),
                Task("quantile_t_test", "Quantile T-Test"),
                Task("causal_impact_analysis", "Causal Impact Analysis"),
            ],
        ),

        # Engineering skills ONLY (python, tests, oop, perf, tooling)
        Domain(
            folder="09_engineering_skills",
            name="Engineering Skills (Python)",
            readme=md_domain_readme(
                summary="Engineering fundamentals for Python: testing, correctness, performance, and maintainable code structure.",
                scope=["Unit/integration testing", "Debugging and edge cases", "Python tooling and code organization", "Performance and refactoring basics"],
                skills=["pytest, mocks, negative testing", "OOP fundamentals and clean APIs", "Config/tooling basics", "Profiling mindset and code hygiene"],
                deliverables=["Runnable tests under tests/", "Clean module structure under src/", "Clear runbook and documented edge cases"],
                dod=["Tests pass locally", "No hardcoded secrets", "Repro steps documented", "Edge cases covered"],
            ),
            tasks=[
                Task("hello_pytest", "Hello Pytest"),
                Task("mock_test", "Mock-Test"),
                Task("negative_tests", "Negative Tests"),
                Task("mutable_word_count", "Mutable Word Count"),
                Task("rotation_bug", "Rotation Bug"),
                Task("smape_debug", "sMAPE (Debug)"),
                Task("docstring_standardization", "Docstring"),
                Task("generators", "Generators"),
                Task("method_decorators", "Method Decorators"),
                Task("map_filter_reduce", "Map, Filter, Reduce"),
                Task("oop_i_class_vs_object", "OOP I: Class vs Object"),
                Task("oop_ii_principles", "OOP II: Principles"),
                Task("async_await", "Async/Await"),
                Task("url_parsing", "URL Parsing"),
                Task("yaml_config", "YAML Config"),
                Task("makefile", "Makefile"),
                Task("multiprocessing", "Multiprocessing"),
                Task("memoization", "Memoization"),
                Task("valid_emails", "Valid Emails"),
                Task("error_analysis", "Error Analysis"),
            ],
        ),

        # Additional domains for remaining tasks
        Domain(
            folder="10_data_platform_mlops",
            name="Data Platform and MLOps",
            readme=md_domain_readme(
                summary="Data quality, platform tooling, and ML lifecycle utilities used in production environments.",
                scope=["Data quality pipelines", "Environment isolation", "Experiment tracking", "Data/model versioning"],
                skills=["PySpark data work", "Docker and compose", "MLflow, DVC", "Operational reliability basics"],
                deliverables=["Runnable pipelines", "Clear run instructions", "Validation checks and artifacts"],
                dod=["Reproducible execution", "Artifacts stored under reports/", "Assumptions documented"],
            ),
            tasks=[
                Task("data_quality_pyspark", "Data Quality & PySpark"),
                Task("hello_docker", "Hello Docker"),
                Task("docker_compose", "Docker-Compose"),
                Task("anti_fraud_i_mlflow", "Anti-Fraud I: MLflow"),
                Task("anti_fraud_ii_dvc", "Anti-Fraud II: DVC"),
                Task("mcp_server", "MCP-Server"),
            ],
        ),

        Domain(
            folder="11_ai_nlp_llm_systems",
            name="AI / NLP / LLM Systems",
            readme=md_domain_readme(
                summary="Applied NLP and LLM system patterns: retrieval augmentation, evaluation, and lightweight applications.",
                scope=["RAG pipelines and evaluation", "LLM-based classification and utilities", "NLP robustness and tokenization basics"],
                skills=["Embedding retrieval patterns", "Chunking and evaluation", "Prompting/application scaffolding", "NLP testing mindset"],
                deliverables=["Runnable demos or pipelines", "Evaluation artifacts", "Clear documentation and runbooks"],
                dod=["Inputs/outputs documented", "Evaluation included where applicable", "Repro steps are complete"],
            ),
            tasks=[
                Task("hello_bert", "Hello BERT"),
                Task("negation", "Negation"),
                Task("video_summary", "Video Summary"),
                Task("rag", "RAG"),
                Task("semantic_chunking", "Semantic Chunking"),
                Task("llm_classification", "LLM Classification"),
                Task("sql_rag", "SQL RAG"),
                Task("second_brain", "Second Brain"),
                Task("rag_eval", "RAG Eval"),
                Task("rag_courses_advisor", "RAG Courses Advisor"),
                Task("hello_gpt2", "Hello, GPT-2"),
                Task("bpe", "BPE"),
                Task("intent_classification", "Intent Classification"),
                Task("visual_intent_classification", "Visual Intent Classification"),
                Task("dwh_assistant", "DWH Assistant"),
                Task("suggest_service", "Suggest Service"),
            ],
        ),

        Domain(
            folder="12_deep_learning_cv",
            name="Deep Learning and Computer Vision",
            readme=md_domain_readme(
                summary="Deep learning and computer vision tasks: detection, metric learning, and deployment-oriented DL patterns.",
                scope=["Object detection and CV pipelines", "Metric learning for identity", "DL training patterns and efficiency"],
                skills=["Model training and evaluation", "Loss functions (Triplet/ArcFace)", "System constraints and performance awareness"],
                deliverables=["Training/evaluation artifacts", "Reproducible runs", "Short technical reports"],
                dod=["Evaluation included", "Artifacts stored under reports/", "Repro steps documented"],
            ),
            tasks=[
                Task("yolo_detection", "YOLO Detection"),
                Task("sam", "SAM"),
                Task("captcha_solver", "Captcha Solver"),
                Task("face_id_i_triplet_loss", "Face ID I: Triplet Loss"),
                Task("face_id_ii_arcface_loss", "Face ID II: ArcFace Loss"),
                Task("parameter_efficient_fine_tuning", "Parameter Efficient Fine-Tuning"),
                Task("model_distillation", "Model Distillation"),
                Task("tasa_transformer", "TASA Transformer"),
                Task("moscow_i", "Moscow I"),
                Task("ocr_eval", "OCR Eval"),
                Task("calorie_tracker", "Calorie Tracker"),
            ],
        ),
    ]

    # Root files
    write_file(repo_root / "README.md", root_readme(domains))
    write_file(repo_root / "PORTFOLIO.md", portfolio_md())
    write_file(repo_root / "catalog.yml", catalog_yaml(domains))
    write_file(repo_root / ".gitignore", gitignore())

    # Create domain + task folders (no "tasks/" folder)
    for d in domains:
        domain_root = repo_root / d.folder
        ensure_dir(domain_root)
        write_file(domain_root / "README.md", d.readme)

        for t in d.tasks:
            make_task_folder(domain_root, d.name, t)

    print(f"Done. Repo scaffold created at: {repo_root}")

if __name__ == "__main__":
    main()

