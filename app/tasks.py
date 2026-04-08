"""
Realistic customer-support ticket dataset and task definitions.

Each task bundles:
  - a ticket (the observable state seed)
  - the expected action sequence (ground-truth for grading)
  - difficulty metadata
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Raw ticket corpus
# ---------------------------------------------------------------------------

TICKETS: List[Dict[str, Any]] = [
    # ── BILLING ──────────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-001",
        "customer_message": (
            "Hi, I was charged twice for my monthly subscription last week. "
            "My bank statement shows two identical charges of $29.99 on the 14th. "
            "I need this fixed immediately."
        ),
        "customer_history": [
            "Account active since 2021-03-10.",
            "Previous refund issued 2022-07-22 for duplicate charge.",
        ],
        "sentiment": "angry",
        "category": "billing",
        "correct_sequence": [
            "classify_billing",
            "issue_refund",
            "close_ticket",
        ],
    },
    {
        "ticket_id": "TKT-002",
        "customer_message": (
            "I think there might be an error in my bill. I was on the basic plan "
            "but the charge looks higher than expected. Can you check?"
        ),
        "customer_history": [
            "Upgraded from Basic to Pro plan on 2024-11-01.",
        ],
        "sentiment": "neutral",
        "category": "billing",
        "correct_sequence": [
            "classify_billing",
            "request_more_info",
            "apply_billing_credit",
            "close_ticket",
        ],
    },

    # ── TECHNICAL ────────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-003",
        "customer_message": (
            "The app keeps crashing every time I try to open the dashboard. "
            "I've tried reinstalling but it still happens. Running iOS 17.4."
        ),
        "customer_history": [
            "Device: iPhone 14 Pro.",
            "App version 3.2.1 installed.",
        ],
        "sentiment": "negative",
        "category": "technical",
        "correct_sequence": [
            "classify_technical",
            "send_technical_guide",
            "restart_service",
            "close_ticket",
        ],
    },
    {
        "ticket_id": "TKT-004",
        "customer_message": (
            "I can't log in. I tried resetting my password but I'm not receiving "
            "the reset email. Please help urgently."
        ),
        "customer_history": [
            "Email on file: john.doe@example.com (verified).",
            "Last login: 2024-12-01.",
        ],
        "sentiment": "negative",
        "category": "technical",
        "correct_sequence": [
            "classify_technical",
            "reset_password",
            "close_ticket",
        ],
    },
    {
        "ticket_id": "TKT-005",
        "customer_message": (
            "Our entire team has been unable to access the API for the past 3 hours. "
            "This is blocking our production deployment. We need immediate resolution."
        ),
        "customer_history": [
            "Enterprise account — 50 seats.",
            "SLA: 1-hour response time.",
            "Account manager: Sarah Collins.",
        ],
        "sentiment": "angry",
        "category": "technical",
        "correct_sequence": [
            "classify_technical",
            "escalate_to_human",
            "restart_service",
            "close_ticket",
        ],
    },

    # ── REFUND ───────────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-006",
        "customer_message": (
            "I'd like to request a refund for the annual subscription I purchased "
            "3 days ago. I realised the plan doesn't include the features I need."
        ),
        "customer_history": [
            "Annual plan purchased 2025-01-10 — $299.",
            "No prior refunds.",
        ],
        "sentiment": "neutral",
        "category": "refund",
        "correct_sequence": [
            "classify_refund",
            "issue_refund",
            "close_ticket",
        ],
    },
    {
        "ticket_id": "TKT-007",
        "customer_message": (
            "I want my money back. Your service has been down for two weeks and "
            "my business has suffered. I'm considering legal action if this isn't "
            "resolved today."
        ),
        "customer_history": [
            "Ongoing outage incident INC-8821 affects this region.",
            "Customer escalated to management 2025-01-08.",
            "Three prior complaint tickets.",
        ],
        "sentiment": "angry",
        "category": "refund",
        "correct_sequence": [
            "classify_refund",
            "escalate_to_human",
            "issue_refund",
            "close_ticket",
        ],
    },

    # ── EDGE CASES (ambiguous) ────────────────────────────────────────────────
    {
        "ticket_id": "TKT-008",
        "customer_message": (
            "Something seems off with my account. I don't know if it's a billing "
            "problem or a technical one but things aren't working right."
        ),
        "customer_history": [],
        "sentiment": "neutral",
        "category": "general",
        "correct_sequence": [
            "classify_general",
            "request_more_info",
            "close_ticket",
        ],
    },

    # ── ADVANCED HARD CASES ────────────────────────────────────────────────
    {
        "ticket_id": "TKT-009",
        "customer_message": (
            "Our enterprise API is timing out for payment endpoints. We have a launch "
            "in 2 hours and legal escalation if SLA is breached."
        ),
        "customer_history": [
            "Enterprise account — platinum tier.",
            "Recent migration completed 48 hours ago.",
            "Prior incident with root cause in rate limiter config.",
        ],
        "sentiment": "angry",
        "category": "technical",
        "correct_sequence": [
            "classify_technical",
            "check_service_status",
            "escalate_to_human",
            "restart_service",
            "close_ticket",
        ],
        "hidden_context": {
            "check_service_status": {
                "facts": [
                    "error_rate=32% on payment endpoints",
                    "blast_radius=high",
                    "primary_failure_mode=rate_limiter_regression",
                ],
                "risk_flags": ["sla_risk_high", "customer_churn_risk"],
            },
            "fetch_customer_history": {
                "facts": [
                    "recent_change_window=true",
                    "prior_incident_pattern=technical_regression",
                ],
                "risk_flags": ["repeat_incident_risk"],
            },
        },
    },
    {
        "ticket_id": "TKT-010",
        "customer_message": (
            "We were charged annual + monthly in the same cycle. Finance is initiating "
            "a chargeback unless this is corrected today."
        ),
        "customer_history": [
            "Billing profile switched from monthly to annual 5 days ago.",
            "Two historical goodwill credits in prior 12 months.",
        ],
        "sentiment": "negative",
        "category": "billing",
        "correct_sequence": [
            "classify_billing",
            "verify_billing_ledger",
            "issue_refund",
            "close_ticket",
        ],
        "hidden_context": {
            "verify_billing_ledger": {
                "facts": [
                    "duplicate_charge_confirmed=true",
                    "charge_age_days=2",
                    "chargeback_probability=high",
                ],
                "risk_flags": ["regulatory_risk", "chargeback_risk"],
            },
            "fetch_customer_history": {
                "facts": [
                    "prior_goodwill_credits=2",
                    "vip_tier_customer=true",
                ],
                "risk_flags": ["retention_risk"],
            },
        },
    },
    {
        "ticket_id": "TKT-011",
        "customer_message": (
            "Our users cannot complete checkout and invoices also look wrong. "
            "Is this a payment bug or billing policy issue?"
        ),
        "customer_history": [
            "Checkout conversion dropped 41% today.",
            "Billing service and payment service deployed independently.",
        ],
        "sentiment": "negative",
        "category": "general",
        "correct_sequence": [
            "classify_general",
            "check_service_status",
            "verify_billing_ledger",
            "request_more_info",
            "close_ticket",
        ],
        "hidden_context": {
            "check_service_status": {
                "facts": [
                    "checkout_timeout_rate=4%",
                    "payment_gateway_status=degraded",
                ],
                "risk_flags": ["technical_uncertainty"],
            },
            "verify_billing_ledger": {
                "facts": [
                    "invoice_rounding_issue_detected=true",
                    "billing_impact_scope=medium",
                ],
                "risk_flags": ["billing_compliance_risk"],
            },
        },
    },
]

# Create a lookup by ticket_id
TICKET_BY_ID: Dict[str, Dict[str, Any]] = {t["ticket_id"]: t for t in TICKETS}


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: List[Dict[str, Any]] = [
    # ── TASK 1: Easy — single-step classification ────────────────────────────
    {
        "task_id": "task_1_classify",
        "difficulty": "easy",
        "description": (
            "Classify the incoming support ticket into the correct category "
            "(billing, technical, refund, or general). "
            "You must issue exactly ONE classification action."
        ),
        "ticket_id": "TKT-001",
        "goal_actions": ["classify_billing"],   # Only the first action is graded
        "max_steps": 3,
        "sla_steps": 3,
    },
    {
        "task_id": "task_1b_classify_technical",
        "difficulty": "easy",
        "description": (
            "Classify a technical issue ticket into the correct category."
        ),
        "ticket_id": "TKT-004",
        "goal_actions": ["classify_technical"],
        "max_steps": 3,
        "sla_steps": 3,
    },
    {
        "task_id": "task_1c_classify_refund",
        "difficulty": "easy",
        "description": (
            "Classify a refund request ticket into the correct category."
        ),
        "ticket_id": "TKT-006",
        "goal_actions": ["classify_refund"],
        "max_steps": 3,
        "sla_steps": 3,
    },

    # ── TASK 2: Medium — choose the correct resolution path ──────────────────
    {
        "task_id": "task_2_resolution_path",
        "difficulty": "medium",
        "description": (
            "Given a pre-classified billing ticket (already marked as billing), "
            "select the correct resolution action sequence: "
            "classify → resolve → close."
        ),
        "ticket_id": "TKT-001",
        "goal_actions": ["classify_billing", "issue_refund", "close_ticket"],
        "max_steps": 5,
        "sla_steps": 5,
    },
    {
        "task_id": "task_2b_technical_path",
        "difficulty": "medium",
        "description": (
            "Resolve a technical login issue: classify → fix → close."
        ),
        "ticket_id": "TKT-004",
        "goal_actions": ["classify_technical", "reset_password", "close_ticket"],
        "max_steps": 5,
        "sla_steps": 5,
    },

    # ── TASK 3: Hard — full multi-step resolution ────────────────────────────
    {
        "task_id": "task_3_multistep_angry_refund",
        "difficulty": "hard",
        "description": (
            "Handle an angry customer demanding a refund after a prolonged outage. "
            "The correct sequence requires classification, escalation, refund, and closure. "
            "Efficiency matters: every unnecessary step reduces your score."
        ),
        "ticket_id": "TKT-007",
        "goal_actions": [
            "classify_refund",
            "escalate_to_human",
            "issue_refund",
            "close_ticket",
        ],
        "max_steps": 8,
        "sla_steps": 6,
    },
    {
        "task_id": "task_3b_multistep_technical_enterprise",
        "difficulty": "hard",
        "description": (
            "Resolve a high-priority enterprise API outage: classify, escalate, "
            "restart service, and close. Penalised for unnecessary steps or SLA breach."
        ),
        "ticket_id": "TKT-005",
        "goal_actions": [
            "classify_technical",
            "escalate_to_human",
            "restart_service",
            "close_ticket",
        ],
        "max_steps": 8,
        "sla_steps": 5,
    },
    {
        "task_id": "task_3c_ambiguous_ticket",
        "difficulty": "hard",
        "description": (
            "Handle an ambiguous ticket where the customer is unsure of the problem. "
            "Correctly classify as general, gather more information, then close cleanly."
        ),
        "ticket_id": "TKT-008",
        "goal_actions": [
            "classify_general",
            "request_more_info",
            "close_ticket",
        ],
        "max_steps": 6,
        "sla_steps": 4,
    },
    {
        "task_id": "task_4_enterprise_rca_tradeoff",
        "difficulty": "hard",
        "description": (
            "High-severity enterprise outage with SLA and legal risk. Agent must gather "
            "evidence before deciding escalation vs immediate recovery actions."
        ),
        "ticket_id": "TKT-009",
        "goal_actions": [
            "classify_technical",
            "check_service_status",
            "escalate_to_human",
            "restart_service",
            "close_ticket",
        ],
        "valid_paths": [
            [
                "classify_technical",
                "check_service_status",
                "escalate_to_human",
                "restart_service",
                "close_ticket",
            ],
            [
                "classify_technical",
                "fetch_customer_history",
                "check_service_status",
                "restart_service",
                "close_ticket",
            ],
        ],
        "required_investigation_actions": [
            "check_service_status",
            "fetch_customer_history",
        ],
        "max_steps": 9,
        "sla_steps": 6,
    },
    {
        "task_id": "task_4b_billing_compliance_tradeoff",
        "difficulty": "hard",
        "description": (
            "Billing dispute with chargeback risk and policy constraints. Agent must verify "
            "ledger evidence and choose refund/credit strategy without exploitative shortcuts."
        ),
        "ticket_id": "TKT-010",
        "goal_actions": [
            "classify_billing",
            "verify_billing_ledger",
            "issue_refund",
            "close_ticket",
        ],
        "valid_paths": [
            [
                "classify_billing",
                "verify_billing_ledger",
                "issue_refund",
                "close_ticket",
            ],
            [
                "classify_billing",
                "fetch_customer_history",
                "verify_billing_ledger",
                "apply_billing_credit",
                "close_ticket",
            ],
        ],
        "required_investigation_actions": [
            "verify_billing_ledger",
            "fetch_customer_history",
        ],
        "max_steps": 8,
        "sla_steps": 6,
    },
    {
        "task_id": "task_4c_ambiguous_multisignal_resolution",
        "difficulty": "hard",
        "description": (
            "Mixed technical and billing symptoms. Agent must resolve ambiguity with "
            "investigation actions before selecting a final policy-safe resolution path."
        ),
        "ticket_id": "TKT-011",
        "goal_actions": [
            "classify_general",
            "check_service_status",
            "verify_billing_ledger",
            "request_more_info",
            "close_ticket",
        ],
        "valid_paths": [
            [
                "classify_general",
                "check_service_status",
                "verify_billing_ledger",
                "request_more_info",
                "close_ticket",
            ],
            [
                "classify_general",
                "verify_billing_ledger",
                "check_service_status",
                "request_more_info",
                "close_ticket",
            ],
        ],
        "required_investigation_actions": [
            "check_service_status",
            "verify_billing_ledger",
        ],
        "max_steps": 9,
        "sla_steps": 7,
    },
]

TASK_BY_ID: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in TASKS}
