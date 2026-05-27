from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import hash_password
from app.models import LegalCategory, User

CATEGORY_SEED = [
    {
        "id": "criminal-law",
        "name": "Criminal Law",
        "law_type": "Criminal Law",
        "ai_law_types": [
            "criminal_law",
            "criminal_procedure",
            "evidence_law",
            "constitutional_law",
        ],
        "description": "Offences, bail, FIRs, arrests, and trial procedure.",
        "icon_key": "gavel",
        "sort_order": 1,
    },
    {
        "id": "property-law",
        "name": "Property Law",
        "law_type": "Property Law",
        "ai_law_types": ["property_law"],
        "description": "Ownership, registration, tenancy, inheritance, and land disputes.",
        "icon_key": "home",
        "sort_order": 2,
    },
    {
        "id": "cyber-law",
        "name": "Cyber Law",
        "law_type": "Cyber Law",
        "ai_law_types": ["cyber_law"],
        "description": "Cybercrime, privacy, data misuse, and digital evidence issues.",
        "icon_key": "shield",
        "sort_order": 3,
    },
    {
        "id": "consumer-rights",
        "name": "Consumer Rights",
        "law_type": "Consumer Rights",
        "ai_law_types": [
            "consumer_rights",
            "contract_law",
            "insurance_law",
        ],
        "description": "Defective goods, unfair trade practices, refunds, and service complaints.",
        "icon_key": "receipt",
        "sort_order": 4,
    },
    {
        "id": "employment-law",
        "name": "Employment Law",
        "law_type": "Employment Law",
        "ai_law_types": ["employment_law"],
        "description": "Employment contracts, workplace disputes, wages, and wrongful termination.",
        "icon_key": "briefcase",
        "sort_order": 5,
    },
    {
        "id": "family-law",
        "name": "Family Law",
        "law_type": "Family Law",
        "ai_law_types": [
            "family_law",
            "tax_law",
            "company_corporate_law",
            "banking_finance_law",
            "environmental_law",
        ],
        "description": "Marriage, divorce, maintenance, custody, and succession questions.",
        "icon_key": "users",
        "sort_order": 6,
    },
]


def seed_baseline_data(db: Session, settings: Settings) -> None:
    now = datetime.now(UTC)
    if settings.seed_reference_data:
        for item in CATEGORY_SEED:
            category = db.get(LegalCategory, item["id"])
            if category is None:
                db.add(
                    LegalCategory(
                        **item,
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
                continue

            category.name = item["name"]
            category.law_type = item["law_type"]
            category.ai_law_types = item["ai_law_types"]
            category.description = item["description"]
            category.icon_key = item["icon_key"]
            category.sort_order = item["sort_order"]
            category.is_active = True
            category.updated_at = now
    if settings.enable_admin_bootstrap:
        admin = db.scalar(select(User).where(User.email == settings.admin_bootstrap_email.lower()))
        if admin is None:
            db.add(
                User(
                    full_name="ILAP Admin",
                    email=settings.admin_bootstrap_email.lower(),
                    password_hash=hash_password(settings.admin_bootstrap_password),
                    role="admin",
                    preferred_practice_areas=[],
                    notification_preferences={
                        "product_updates": True,
                        "support_followups": True,
                        "security_alerts": True,
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
    db.commit()
