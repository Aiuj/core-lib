# 🏭 Industry Categories

The `core-lib` provides a comprehensive, multilingual taxonomy of industry categories used across all Faciliter applications.

## Overview

Industry categories enable consistent classification of companies across the platform with:
- **28 standardized categories** organized into 7 broad areas
- **Multilingual support** (English, French, German) via gettext
- **Material Design icons** for visual consistency
- **Detailed descriptions** for accurate classification

## Usage

### Basic Usage

```python
from core_lib.config.industry_categories import (
    INDUSTRY_CATEGORIES,
    INDUSTRY_CATEGORIES_BY_KEY,
    INDUSTRY_CATEGORY_CHOICES
)

# Get all categories
categories = INDUSTRY_CATEGORIES

# Fast lookup by key
category = INDUSTRY_CATEGORIES_BY_KEY['tech_software']
print(category['label'])  # "Software & IT Services"
print(category['area'])   # "TECHNOLOGY"

# Django form choices
form_field = forms.ChoiceField(choices=INDUSTRY_CATEGORY_CHOICES)
```

### Multilingual Support

Labels are wrapped with gettext `_()` for automatic translation:

```python
from django.utils.translation import activate
from core_lib.config.industry_categories import INDUSTRY_CATEGORIES_BY_KEY

# English
activate('en')
print(INDUSTRY_CATEGORIES_BY_KEY['tech_software']['label'])
# Output: "Software & IT Services"

# French
activate('fr')
print(INDUSTRY_CATEGORIES_BY_KEY['tech_software']['label'])
# Output: "Logiciels et services informatiques"

# German
activate('de')
print(INDUSTRY_CATEGORIES_BY_KEY['tech_software']['label'])
# Output: "Software und IT-Dienstleistungen"
```

## Category Structure

Each category includes:

| Field | Type | Description |
|-------|------|-------------|
| `key` | str | Unique identifier (e.g., `tech_software`) |
| `label` | str | Translatable display name |
| `description` | str | Detailed explanation |
| `icon` | str | Material Design icon name |
| `area` | str | Broader sector grouping |

## Available Categories

### 🖥️ TECHNOLOGY
- **tech_software** - Software & IT Services
- **tech_hardware** - Hardware & Electronics
- **tech_telecom** - Telecommunications

### 🏭 MANUFACTURING
- **mfg_automotive** - Automotive
- **mfg_machinery** - Industrial Machinery
- **mfg_chemicals** - Chemicals & Materials
- **mfg_food_bev** - Food & Beverage
- **mfg_textiles** - Textiles & Apparel

### 🏥 HEALTHCARE
- **health_pharma** - Pharmaceuticals
- **health_devices** - Medical Devices
- **health_services** - Healthcare Services

### 💰 FINANCE
- **fin_banking** - Banking & Lending
- **fin_insurance** - Insurance
- **fin_investment** - Investment & Asset Management

### 🛍️ RETAIL
- **retail_ecommerce** - E-commerce
- **retail_store** - Brick & Mortar Retail
- **wholesale_dist** - Wholesale Distribution

### 🏗️ CONSTRUCTION
- **const_residential** - Residential Construction
- **const_commercial** - Commercial Construction
- **real_estate** - Real Estate Services

### 🚚 LOGISTICS
- **log_freight** - Freight & Shipping
- **log_warehousing** - Warehousing & Storage

### ⚡ ENERGY
- **energy_oil_gas** - Oil & Gas
- **energy_renewables** - Renewable Energy
- **energy_utilities** - Utilities

### 🎯 SERVICES
- **serv_legal** - Legal Services
- **serv_consulting** - Management Consulting
- **serv_marketing** - Marketing & Advertising

## Django Integration

### Model Field

Use as a CharField with choices:

```python
from django.db import models
from core_lib.config.industry_categories import INDUSTRY_CATEGORY_CHOICES

class Company(models.Model):
    industry_category = models.CharField(
        max_length=50,
        choices=INDUSTRY_CATEGORY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Industry Category"
    )
    
    @property
    def industry_category_label(self):
        """Get translated label dynamically"""
        from core_lib.config.industry_categories import INDUSTRY_CATEGORIES_BY_KEY
        return INDUSTRY_CATEGORIES_BY_KEY[self.industry_category]['label']
    
    @property
    def industry_area(self):
        """Get industry area"""
        from core_lib.config.industry_categories import INDUSTRY_CATEGORIES_BY_KEY
        return INDUSTRY_CATEGORIES_BY_KEY[self.industry_category]['area']
```

### Admin Display

```python
from django.contrib import admin
from core_lib.config.industry_categories import INDUSTRY_CATEGORIES_BY_KEY

class CompanyAdmin(admin.ModelAdmin):
    list_display = ['name', 'industry_display']
    
    def industry_display(self, obj):
        """Show translated industry label"""
        if not obj.industry_category:
            return '-'
        cat = INDUSTRY_CATEGORIES_BY_KEY.get(obj.industry_category, {})
        return cat.get('label', obj.industry_category)
```

## Adding Translations

To add or update translations:

```bash
# 1. Extract translatable strings
cd core-lib
python -m babel extract -F babel.cfg -o core_lib/locale/messages.pot .

# 2. Update existing translations
python -m babel update -i core_lib/locale/messages.pot -d core_lib/locale

# 3. Edit .po files
# Edit core_lib/locale/fr/LC_MESSAGES/django.po
# Edit core_lib/locale/de/LC_MESSAGES/django.po

# 4. Compile translations
python -m babel compile -d core_lib/locale
```

Or using Django tools:

```bash
# Generate .po files
django-admin makemessages -l fr -l de --pythonpath=core_lib

# Compile .mo files
django-admin compilemessages --pythonpath=core_lib
```

## Best Practices

### ✅ DO
- Store only the **key** in the database (e.g., `tech_software`)
- Compute **label** and **area** dynamically via properties
- Use `INDUSTRY_CATEGORIES_BY_KEY` for lookups
- Let gettext handle translations automatically

### ❌ DON'T
- Store translated labels in the database
- Hardcode industry names in code
- Create custom industry taxonomies
- Mix keys and labels

## API Response Example

```json
{
  "industry_category": "tech_software",
  "industry_category_label": "Software & IT Services",
  "industry_area": "TECHNOLOGY"
}
```

The label will automatically translate based on the `Accept-Language` header.

## Future Expansion

To add new categories:

1. Add to `INDUSTRY_CATEGORIES` in `core_lib/config/industry_categories.py`
2. Extract strings: `python -m babel extract ...`
3. Update translations: Add to `.po` files
4. Compile: `python -m babel compile ...`

Ensure new categories follow the naming convention: `{area}_{category}` (e.g., `tech_ai`, `health_lab`).
