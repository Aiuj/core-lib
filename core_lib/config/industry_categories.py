try:
    from gettext import gettext as _  # type: ignore
except Exception:  # pragma: no cover - fallback safety
    def _(s: str) -> str:  # type: ignore
        return s

INDUSTRY_CATEGORIES = [
    # Technology
    {
        "key": "tech_software",
        "label": _("Software & IT Services"),
        "description": _("Software development, SaaS, cloud computing, and IT consulting services."),
        "icon": "code",
        "area": "TECHNOLOGY",
    },
    {
        "key": "tech_hardware",
        "label": _("Hardware & Electronics"),
        "description": _("Computer hardware, semiconductors, consumer electronics, and components."),
        "icon": "memory",
        "area": "TECHNOLOGY",
    },
    {
        "key": "tech_telecom",
        "label": _("Telecommunications"),
        "description": _("Wireless, wireline, ISP, and network infrastructure services."),
        "icon": "cell_tower",
        "area": "TECHNOLOGY",
    },

    # Manufacturing
    {
        "key": "mfg_automotive",
        "label": _("Automotive"),
        "description": _("Vehicle manufacturing, parts, and automotive technology."),
        "icon": "directions_car",
        "area": "MANUFACTURING",
    },
    {
        "key": "mfg_machinery",
        "label": _("Industrial Machinery"),
        "description": _("Heavy equipment, industrial tools, and automation systems."),
        "icon": "precision_manufacturing",
        "area": "MANUFACTURING",
    },
    {
        "key": "mfg_chemicals",
        "label": _("Chemicals & Materials"),
        "description": _("Basic chemicals, specialty materials, plastics, and raw materials."),
        "icon": "science",
        "area": "MANUFACTURING",
    },
    {
        "key": "mfg_food_bev",
        "label": _("Food & Beverage"),
        "description": _("Food processing, beverage production, and packaging."),
        "icon": "restaurant",
        "area": "MANUFACTURING",
    },
    {
        "key": "mfg_textiles",
        "label": _("Textiles & Apparel"),
        "description": _("Clothing, fabrics, footwear, and fashion accessories."),
        "icon": "checkroom",
        "area": "MANUFACTURING",
    },

    # Healthcare
    {
        "key": "health_pharma",
        "label": _("Pharmaceuticals"),
        "description": _("Drug development, biotechnology, and pharmaceutical manufacturing."),
        "icon": "medication",
        "area": "HEALTHCARE",
    },
    {
        "key": "health_devices",
        "label": _("Medical Devices"),
        "description": _("Medical equipment, diagnostic tools, and health technology."),
        "icon": "medical_services",
        "area": "HEALTHCARE",
    },
    {
        "key": "health_services",
        "label": _("Healthcare Services"),
        "description": _("Hospitals, clinics, care facilities, and health management."),
        "icon": "local_hospital",
        "area": "HEALTHCARE",
    },

    # Finance
    {
        "key": "fin_banking",
        "label": _("Banking & Lending"),
        "description": _("Commercial banking, loans, credit services, and fintech."),
        "icon": "account_balance",
        "area": "FINANCE",
    },
    {
        "key": "fin_insurance",
        "label": _("Insurance"),
        "description": _("Life, property, casualty, and health insurance services."),
        "icon": "security",
        "area": "FINANCE",
    },
    {
        "key": "fin_investment",
        "label": _("Investment & Asset Mgmt"),
        "description": _("Wealth management, venture capital, and private equity."),
        "icon": "trending_up",
        "area": "FINANCE",
    },

    # Retail & Wholesale
    {
        "key": "retail_ecommerce",
        "label": _("E-commerce"),
        "description": _("Online retail, marketplaces, and digital commerce platforms."),
        "icon": "shopping_cart",
        "area": "RETAIL",
    },
    {
        "key": "retail_store",
        "label": _("Brick & Mortar Retail"),
        "description": _("Physical stores, supermarkets, and department stores."),
        "icon": "storefront",
        "area": "RETAIL",
    },
    {
        "key": "wholesale_dist",
        "label": _("Wholesale Distribution"),
        "description": _("B2B distribution, supply chain intermediaries, and bulk trading."),
        "icon": "warehouse",
        "area": "RETAIL",
    },

    # Construction & Real Estate
    {
        "key": "const_residential",
        "label": _("Residential Construction"),
        "description": _("Home building, renovation, and residential development."),
        "icon": "home",
        "area": "CONSTRUCTION",
    },
    {
        "key": "const_commercial",
        "label": _("Commercial Construction"),
        "description": _("Office buildings, industrial facilities, and public infrastructure."),
        "icon": "apartment",
        "area": "CONSTRUCTION",
    },
    {
        "key": "real_estate",
        "label": _("Real Estate Services"),
        "description": _("Property management, brokerage, and real estate investment."),
        "icon": "real_estate_agent",
        "area": "CONSTRUCTION",
    },

    # Logistics & Transport
    {
        "key": "log_freight",
        "label": _("Freight & Shipping"),
        "description": _("Air, sea, rail, and road freight transportation."),
        "icon": "local_shipping",
        "area": "LOGISTICS",
    },
    {
        "key": "log_warehousing",
        "label": _("Warehousing & Storage"),
        "description": _("Inventory storage, fulfillment centers, and logistics hubs."),
        "icon": "inventory_2",
        "area": "LOGISTICS",
    },

    # Energy & Utilities
    {
        "key": "energy_oil_gas",
        "label": _("Oil & Gas"),
        "description": _("Exploration, extraction, refining, and distribution."),
        "icon": "oil_barrel",
        "area": "ENERGY",
    },
    {
        "key": "energy_renewables",
        "label": _("Renewable Energy"),
        "description": _("Solar, wind, hydro, and sustainable energy solutions."),
        "icon": "solar_power",
        "area": "ENERGY",
    },
    {
        "key": "energy_utilities",
        "label": _("Utilities"),
        "description": _("Water, electricity, waste management, and public services."),
        "icon": "water_drop",
        "area": "ENERGY",
    },

    # Professional Services
    {
        "key": "serv_legal",
        "label": _("Legal Services"),
        "description": _("Law firms, legal consulting, and compliance services."),
        "icon": "gavel",
        "area": "SERVICES",
    },
    {
        "key": "serv_consulting",
        "label": _("Management Consulting"),
        "description": _("Business strategy, operations, and HR consulting."),
        "icon": "groups",
        "area": "SERVICES",
    },
    {
        "key": "serv_marketing",
        "label": _("Marketing & Advertising"),
        "description": _("Digital marketing, branding, PR, and media services."),
        "icon": "campaign",
        "area": "SERVICES",
    },
]

# Fast lookup by key
INDUSTRY_CATEGORIES_BY_KEY = {c["key"]: c for c in INDUSTRY_CATEGORIES}

# Choices helpers for forms/admin
INDUSTRY_CATEGORY_CHOICES = [(c["key"], c["label"]) for c in INDUSTRY_CATEGORIES]
