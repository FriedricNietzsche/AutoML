# 🚀 Quick Start Guide - Redaction System

## Installation

### Step 1: Install Dependencies

```bash
cd backend
pip install presidio-analyzer presidio-anonymizer
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

**Note**: On first run, Presidio will download NLP models (~50MB). This happens automatically.

### Step 2: Start the Backend

```bash
# From backend directory
uvicorn app.main:app --reload --port 8000
```

You should see:
```
[Startup] ✅ Redaction enabled - All logs will be automatically sanitized
✅ Redaction middleware enabled - All API responses will be sanitized
```

If you see warnings about Presidio, make sure you installed it (Step 1).

### Step 3: Test Redaction (Optional)

```bash
# From backend directory
python test_redaction.py
```

Expected output:
```
🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒
REDACTION SYSTEM TEST SUITE
🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒🔒

✅ Presidio is available and loaded
...
🎉 All tests passed!
```

## Verification

### Check Logs

Upload a CSV with sample data containing:
- Emails (e.g., john@example.com)
- Phone numbers (e.g., 555-1234)
- Names (e.g., John Doe)

**Before Redaction:**
```
INFO - User chat: Contact john@example.com
```

**After Redaction:**
```
INFO - User chat: Contact ********
```

### Check API Responses

Call any API endpoint that returns user data. All PII should be masked:

```bash
curl http://localhost:8000/api/projects/demo/data
```

Response (redacted):
```json
{
  "email": "********",
  "phone": "********",
  "name": "********"
}
```

### Check Data Files

Look at saved CSV samples in `backend/assets/projects/*/sample.csv`:
- Should show redacted data
- Original data stays in your database (only samples are redacted)

## Troubleshooting

### Presidio Not Installing

If you get errors installing Presidio:

```bash
# Try updating pip first
python -m pip install --upgrade pip

# Install Presidio components separately
pip install presidio-analyzer
pip install presidio-anonymizer

# If still failing, install spaCy model manually
python -m spacy download en_core_web_lg
```

### Redaction Not Working

Check console output on startup:
- ✅ Good: "Redaction enabled"
- ⚠️ Bad: "Redaction disabled - Presidio not installed"

Verify in Python:
```python
from app.utils import PRESIDIO_AVAILABLE
print(f"Presidio available: {PRESIDIO_AVAILABLE}")
```

### Performance Issues

If redaction is too slow:

1. **Increase threshold** (less sensitive, faster):
   ```python
   # In app/main.py
   redaction_config = RedactionConfig(
       score_threshold=0.5  # Default is 0.35
   )
   ```

2. **Disable for specific endpoints**:
   ```python
   # In app/middleware/redaction.py
   skip_paths = ['/api/some-endpoint']
   ```

3. **Disable middleware** (keep logging redaction only):
   ```python
   # In app/main.py
   configure_redaction_middleware(app, enabled=False)
   ```

## Configuration

Edit `app/main.py` to customize redaction:

```python
redaction_config = RedactionConfig(
    enabled=True,              # Master switch
    score_threshold=0.35,      # Lower = more sensitive
    show_entity_type=False,    # True for debugging
    mask_char="*",             # Character for masking
    hash_sensitive_data=False, # True for consistent hashing
)
```

## What's Protected

✅ **Logs** - All console/file logs automatically redacted
✅ **API Responses** - All JSON responses automatically redacted  
✅ **CSV Samples** - Data samples saved to disk are redacted
✅ **DataFrames** - Pandas data automatically scanned and redacted

## What's NOT Redacted

⚠️ **Original uploaded files** - Stored as-is (only samples are redacted)
⚠️ **Database** - Direct DB queries (redaction is on output only)
⚠️ **Internal variables** - Only output is redacted, not in-memory data

## Security Note

Redaction is a **defense layer**, not a substitute for:
- Proper access controls
- Encryption at rest
- Secure data collection practices
- GDPR/CCPA/HIPAA compliance

Always minimize PII collection!

## Need Help?

- 📖 **Full Docs**: See `REDACTION_SYSTEM.md`
- 🧪 **Tests**: Run `python test_redaction.py`
- 🐛 **Issues**: Check logs for error messages
- 📚 **Presidio Docs**: https://microsoft.github.io/presidio/

---

**Ready to use!** 🎉

The system is now protecting your logs and API responses automatically.
