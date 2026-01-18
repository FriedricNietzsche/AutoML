# Data Redaction System

## Overview

The AutoML backend implements **comprehensive, multi-layered data redaction** using **Microsoft Presidio**, an enterprise-grade PII (Personally Identifiable Information) detection and anonymization framework. This ensures that sensitive data is **never exposed** in logs, API responses, or data samples **by design**.

## 🔒 Security by Design

### Three Layers of Protection

1. **Logging Redaction** - All log output is automatically sanitized
2. **Data Redaction** - DataFrames and data samples are redacted before storage/transmission  
3. **API Response Redaction** - All API responses are automatically sanitized via middleware

### Automatic Detection

The system automatically detects and redacts:

- ✅ **Email addresses**
- ✅ **Phone numbers** (US and international formats)
- ✅ **Social Security Numbers (SSN)**
- ✅ **Credit card numbers**
- ✅ **Person names**
- ✅ **Physical locations/addresses**
- ✅ **Dates of birth**
- ✅ **IP addresses** (IPv4 and IPv6)
- ✅ **URLs with sensitive parameters**
- ✅ **Medical license numbers**
- ✅ **Crypto wallet addresses**
- ✅ **Bank account numbers** (US, IBAN)
- ✅ **Driver's licenses, passports**
- ✅ **JWT tokens, API keys**
- ✅ **And 20+ more PII types...**

## 🚀 How It Works

### 1. Logging Redaction

All loggers are automatically configured with redaction on startup:

```python
# From app/main.py
from app.utils import setup_redacted_logging, RedactionConfig

redaction_config = RedactionConfig(
    enabled=True,
    score_threshold=0.35,  # Confidence threshold
)

setup_redacted_logging(config=redaction_config)
```

**Before:**
```
INFO - User chat: My email is john.doe@example.com and SSN is 123-45-6789
```

**After:**
```
INFO - User chat: My email is ******** and SSN is ***-**-6789
```

### 2. Data Redaction

Data is redacted before being saved or sent to clients:

```python
# From app/api/data.py
from app.utils import redact_dataframe, DATA_REDACTION_CONFIG

# Redact DataFrame before saving sample
df_redacted = redact_dataframe(df, config=DATA_REDACTION_CONFIG)
df_redacted.to_csv(sample_path, index=False)
```

**DataFrame Before:**
```
name          email                  phone         ssn
John Doe      john@example.com       555-1234      123-45-6789
Jane Smith    jane@company.org       555-5678      987-65-4321
```

**DataFrame After:**
```
name          email                  phone         ssn
********      ********************   ********      ***-**-6789
********      ********************   ********      ***-**-4321
```

### 3. API Response Redaction

Middleware automatically redacts all JSON responses:

```python
# From app/main.py
from app.middleware import configure_redaction_middleware

configure_redaction_middleware(app, enabled=True)
```

**API Response Before:**
```json
{
  "user": "John Doe",
  "email": "john@example.com",
  "phone": "555-1234"
}
```

**API Response After:**
```json
{
  "user": "********",
  "email": "********************",
  "phone": "********"
}
```

## 📦 Installation

The redaction system requires Presidio:

```bash
pip install presidio-analyzer presidio-anonymizer
```

This is automatically included in `requirements.txt`.

### First-Time Setup

Presidio downloads language models on first use:

```python
# This happens automatically on first import
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()  # Downloads spaCy model (~50MB)
```

## ⚙️ Configuration

### RedactionConfig Options

```python
from app.utils import RedactionConfig

config = RedactionConfig(
    # Core settings
    enabled=True,              # Master switch
    language="en",             # Language for PII detection
    score_threshold=0.35,      # Confidence threshold (0-1)
    
    # Redaction style
    mask_char="*",             # Character for masking
    hash_sensitive_data=False, # Use hash instead of mask
    show_entity_type=False,    # Show [EMAIL], [PHONE] labels
    
    # Entity selection
    entities=None,             # None = all entities, or list specific types
    
    # Whitelisting
    whitelist_columns={'id', 'timestamp'},  # Never redact these columns
)
```

### Environment-Specific Configuration

```python
# Development: Show entity types for debugging
dev_config = RedactionConfig(
    show_entity_type=True,
    score_threshold=0.3,  # More sensitive
)

# Production: Maximum protection
prod_config = RedactionConfig(
    enabled=True,
    score_threshold=0.35,
    hash_sensitive_data=True,  # Consistent redaction
)
```

## 🧪 Testing Redaction

### Manual Testing

```python
from app.utils import redact_string, redact_dict, redact_dataframe
import pandas as pd

# Test string redaction
text = "Contact me at john@example.com or call 555-1234"
print(redact_string(text))
# Output: "Contact me at ******** or call ********"

# Test dict redaction
data = {
    "name": "John Doe",
    "email": "john@example.com",
    "api_key": "sk_live_abc123xyz789"
}
print(redact_dict(data))
# Output: {"name": "********", "email": "********", "api_key": "[REDACTED]"}

# Test DataFrame redaction
df = pd.DataFrame({
    "name": ["John Doe", "Jane Smith"],
    "email": ["john@example.com", "jane@company.org"]
})
print(redact_dataframe(df))
```

### Unit Tests

```bash
cd backend
python -m app.utils.redaction
# Runs built-in tests
```

## 🎯 Usage Examples

### Custom Redaction in Endpoints

```python
from app.utils import redact_dict, RedactionConfig

@router.get("/sensitive-data")
async def get_sensitive_data():
    # Your data
    data = fetch_data_from_db()
    
    # Custom redaction config for this endpoint
    config = RedactionConfig(
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],  # Only these
        score_threshold=0.5,  # Higher threshold
    )
    
    # Redact before returning
    return redact_dict(data, config)
```

### Conditional Redaction

```python
from app.utils import redact_for_display, PRESIDIO_AVAILABLE

def get_user_data(admin: bool = False):
    data = fetch_user_data()
    
    # Only redact for non-admin users
    if not admin:
        data = redact_for_display(data)
    
    return data
```

### Whitelisting Columns

```python
from app.utils import redact_dataframe, RedactionConfig

# Never redact these columns
config = RedactionConfig(
    whitelist_columns={'user_id', 'timestamp', 'category'}
)

df_safe = redact_dataframe(df, config)
```

## 📊 Performance

Presidio is optimized but adds some overhead:

- **String redaction**: ~5-20ms per string (cached model)
- **DataFrame redaction**: ~50-200ms for 500 rows
- **API middleware**: ~10-50ms per response

### Optimization Tips

1. **Whitelist non-sensitive endpoints** in middleware
2. **Cache redacted results** when possible
3. **Use score_threshold** to tune sensitivity vs. speed
4. **Redact at source** (once) rather than multiple times

## 🔧 Troubleshooting

### Presidio Not Installing

```bash
# Install with specific versions
pip install presidio-analyzer==2.2.0 presidio-anonymizer==2.2.0

# Download spaCy model manually if needed
python -m spacy download en_core_web_lg
```

### Redaction Not Working

Check logs for:
```
[Startup] ⚠️  Redaction disabled - Presidio not installed
```

Verify installation:
```python
from app.utils import PRESIDIO_AVAILABLE
print(f"Presidio available: {PRESIDIO_AVAILABLE}")
```

### False Positives

Adjust confidence threshold:
```python
# Less sensitive (fewer false positives)
config = RedactionConfig(score_threshold=0.5)

# More sensitive (more false positives)
config = RedactionConfig(score_threshold=0.2)
```

### Custom Patterns

Add custom entity recognizers:
```python
from presidio_analyzer import Pattern, PatternRecognizer

# Define custom pattern
custom_recognizer = PatternRecognizer(
    supported_entity="CUSTOM_ID",
    patterns=[Pattern("custom_id", r"CID-\d{8}", 0.9)]
)

# Add to analyzer
analyzer.registry.add_recognizer(custom_recognizer)
```

## 🛡️ Security Best Practices

1. **✅ Always enable redaction in production**
2. **✅ Review logs periodically for leaks**
3. **✅ Test with real-world PII patterns**
4. **✅ Whitelist only truly non-sensitive columns**
5. **✅ Use HTTPS for all API communications**
6. **✅ Rotate API keys and secrets regularly**
7. **⚠️ Never disable redaction for debugging in prod**
8. **⚠️ Don't log raw user input without redaction**

## 📚 Additional Resources

- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [Supported Entities](https://microsoft.github.io/presidio/supported_entities/)
- [Custom Recognizers](https://microsoft.github.io/presidio/tutorial/08_no_code/)
- [Best Practices](https://microsoft.github.io/presidio/best_practices/)

## 🤝 Contributing

To add new redaction features:

1. Update `app/utils/redaction.py`
2. Add tests to `_test_redaction()`
3. Document in this README
4. Test with real PII examples

## 📝 License

This redaction system uses:
- **Presidio** (MIT License) by Microsoft
- **spaCy** (MIT License) for NLP

---

**Remember**: Redaction is a last line of defense. Always minimize PII collection and follow data protection regulations (GDPR, CCPA, HIPAA, etc.).
