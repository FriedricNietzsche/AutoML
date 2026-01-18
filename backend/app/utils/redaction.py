"""
Data redaction utilities using Microsoft Presidio for sensitive information protection.

This module provides comprehensive redaction capabilities for:
- Logging output (via custom formatter)
- Data processing (pandas DataFrames, dicts, strings)
- API responses

Sensitive patterns automatically detected and redacted by Presidio:
- Email addresses
- Phone numbers
- Credit card numbers
- Social Security Numbers (SSN)
- Person names
- Locations
- Dates of birth
- IP addresses
- URLs
- Medical license numbers
- Crypto wallet addresses
- And many more...

Dependencies:
    pip install presidio-analyzer presidio-anonymizer
"""
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Union
import json

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("[WARNING] Presidio not installed. Install with: pip install presidio-analyzer presidio-anonymizer")
    print("[WARNING] Redaction will be DISABLED until Presidio is installed.")


# ============================================================================
# REDACTION CONFIGURATION
# ============================================================================

class RedactionConfig:
    """Configuration for redaction behavior."""
    
    def __init__(
        self,
        enabled: bool = True,
        language: str = "en",
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.35,
        mask_char: str = "*",
        hash_sensitive_data: bool = False,
        show_entity_type: bool = False,  # Show what type was redacted, e.g., [EMAIL]
        whitelist_columns: Optional[Set[str]] = None,
    ):
        """
        Initialize redaction configuration.
        
        Args:
            enabled: Whether redaction is enabled
            language: Language for PII detection (default: "en")
            entities: List of entity types to redact. If None, uses default comprehensive list.
                     Available entities: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
                     SSN, IBAN_CODE, IP_ADDRESS, LOCATION, DATE_TIME, MEDICAL_LICENSE,
                     URL, CRYPTO, and many more.
            score_threshold: Minimum confidence score (0-1) for detection
            mask_char: Character to use for masking
            hash_sensitive_data: Use hash instead of masking
            show_entity_type: Show entity type in redaction, e.g., [EMAIL_ADDRESS]
            whitelist_columns: Column names to never redact
        """
        self.enabled = enabled and PRESIDIO_AVAILABLE
        self.language = language
        
        # Default comprehensive entity list
        if entities is None:
            self.entities = [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "US_DRIVER_LICENSE",
                "US_PASSPORT",
                "US_BANK_NUMBER",
                "IBAN_CODE",
                "IP_ADDRESS",
                "LOCATION",
                "DATE_TIME",
                "MEDICAL_LICENSE",
                "URL",
                "CRYPTO",
                "UK_NHS",
                "SG_NRIC_FIN",
                "AU_ABN",
                "AU_ACN",
                "AU_TFN",
                "AU_MEDICARE",
            ]
        else:
            self.entities = entities
            
        self.score_threshold = score_threshold
        self.mask_char = mask_char
        self.hash_sensitive_data = hash_sensitive_data
        self.show_entity_type = show_entity_type
        self.whitelist_columns = whitelist_columns or set()
        
        # Initialize Presidio engines if available
        if PRESIDIO_AVAILABLE and self.enabled:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None


# Default global configuration
DEFAULT_CONFIG = RedactionConfig()


# ============================================================================
# REDACTION FUNCTIONS
# ============================================================================

def redact_string(text: str, config: RedactionConfig = DEFAULT_CONFIG) -> str:
    """
    Redact sensitive information from a string using Presidio.
    
    Args:
        text: Input string to redact
        config: Redaction configuration
        
    Returns:
        Redacted string
    """
    if not config.enabled or not text or not config.analyzer:
        return text
    
    try:
        # Analyze the text for PII
        analyzer_results = config.analyzer.analyze(
            text=text,
            entities=config.entities,
            language=config.language,
            score_threshold=config.score_threshold,
        )
        
        # Anonymize the detected PII
        if config.hash_sensitive_data:
            # Use hash for consistent redaction
            operators = {
                entity_type: OperatorConfig("hash")
                for entity_type in config.entities
            }
        elif config.show_entity_type:
            # Replace with entity type label
            operators = {
                entity_type: OperatorConfig("replace", {"new_value": f"[{entity_type}]"})
                for entity_type in config.entities
            }
        else:
            # Use mask (default)
            operators = {
                entity_type: OperatorConfig("mask", {
                    "masking_char": config.mask_char,
                    "chars_to_mask": 100,
                    "from_end": False
                })
                for entity_type in config.entities
            }
        
        anonymized_result = config.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )
        
        return anonymized_result.text
    except Exception as e:
        # If redaction fails, log the error but return original text
        # to avoid breaking the application
        logging.getLogger(__name__).warning(f"Redaction failed: {e}")
        return text


def redact_dict(
    data: Dict[str, Any],
    config: RedactionConfig = DEFAULT_CONFIG,
    sensitive_keys: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Redact sensitive information from a dictionary.
    
    Args:
        data: Dictionary to redact
        config: Redaction configuration
        sensitive_keys: Additional keys to always redact completely
        
    Returns:
        Redacted dictionary
    """
    if not config.enabled:
        return data
    
    # Keys that should be completely redacted (not analyzed)
    always_redact_keys = {
        'password', 'passwd', 'pwd',
        'secret', 'api_key', 'apikey', 'token', 'bearer',
        'access_token', 'refresh_token',
        'private_key', 'ssh_key', 'api_secret',
        'client_secret', 'auth_token',
    }
    
    if sensitive_keys:
        always_redact_keys.update(sensitive_keys)
    
    result = {}
    for key, value in data.items():
        # Check if key is in whitelist
        if key in config.whitelist_columns:
            result[key] = value
            continue
        
        # Check if key should be completely redacted
        if any(s in key.lower() for s in always_redact_keys):
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact_string(value, config)
        elif isinstance(value, dict):
            result[key] = redact_dict(value, config, sensitive_keys)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item, config, sensitive_keys) if isinstance(item, dict)
                else redact_string(item, config) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


def redact_dataframe(
    df: pd.DataFrame,
    config: RedactionConfig = DEFAULT_CONFIG,
    sensitive_columns: Optional[List[str]] = None,
    auto_detect: bool = True,
) -> pd.DataFrame:
    """
    Redact sensitive information from a pandas DataFrame.
    
    This creates a copy of the DataFrame with sensitive data redacted.
    Original DataFrame is not modified.
    
    Args:
        df: DataFrame to redact
        config: Redaction configuration
        sensitive_columns: Explicitly list columns that contain sensitive data
        auto_detect: Whether to auto-detect sensitive columns by name
        
    Returns:
        Redacted DataFrame (copy)
    """
    if not config.enabled or df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    result = df.copy()
    
    columns_to_redact = set()
    
    # Add explicitly marked columns
    if sensitive_columns:
        columns_to_redact.update(sensitive_columns)
    
    # Auto-detect based on column names
    if auto_detect:
        potential_sensitive_patterns = [
            'email', 'phone', 'ssn', 'social',
            'password', 'passwd', 'pwd',
            'api_key', 'token', 'secret',
            'credit', 'card', 'address',
            'dob', 'birth', 'name', 'firstname', 'lastname',
            'ip', 'ip_address', 'location',
        ]
        
        for col in result.columns:
            col_lower = str(col).lower()
            if col in config.whitelist_columns:
                continue
            if any(pattern in col_lower for pattern in potential_sensitive_patterns):
                columns_to_redact.add(col)
    
    # Redact identified columns
    for col in columns_to_redact:
        if col in result.columns:
            # Only process string/object columns
            if result[col].dtype == 'object':
                result[col] = result[col].apply(
                    lambda x: redact_string(str(x), config) if pd.notna(x) else x
                )
    
    # Also scan all text columns for PII even if not in sensitive list
    # This catches PII that might be in unexpected columns
    if config.enabled and auto_detect:
        for col in result.columns:
            if col in config.whitelist_columns or col in columns_to_redact:
                continue
            
            if result[col].dtype == 'object':
                # Sample first few rows to check for PII
                sample = result[col].head(10).dropna()
                if len(sample) > 0:
                    sample_text = ' '.join(sample.astype(str).tolist())
                    redacted_sample = redact_string(sample_text, config)
                    
                    # If redaction changed the text, this column likely has PII
                    if sample_text != redacted_sample:
                        result[col] = result[col].apply(
                            lambda x: redact_string(str(x), config) if pd.notna(x) else x
                        )
    
    return result


# ============================================================================
# LOGGING INTEGRATION
# ============================================================================

class RedactingFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive information using Presidio.
    
    Use this formatter with any logging handler to automatically
    redact sensitive data from all log output.
    
    Example:
        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger = logging.getLogger('my_app')
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        config: RedactionConfig = DEFAULT_CONFIG
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.config = config
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with redaction applied."""
        # First, format the message normally
        original = super().format(record)
        
        # Then redact sensitive information
        if self.config.enabled:
            return redact_string(original, self.config)
        
        return original


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_redacted_logging(
    logger: Optional[logging.Logger] = None,
    config: RedactionConfig = DEFAULT_CONFIG
) -> None:
    """
    Configure a logger to use redacting formatter for all handlers.
    
    Args:
        logger: Logger to configure (if None, configures root logger)
        config: Redaction configuration
    """
    if not config.enabled:
        logging.warning("Redaction is disabled (Presidio not available or disabled in config)")
        return
    
    target_logger = logger or logging.getLogger()
    
    # Update all existing handlers
    for handler in target_logger.handlers:
        current_formatter = handler.formatter
        if current_formatter:
            # Preserve existing format string
            fmt = current_formatter._fmt if hasattr(current_formatter, '_fmt') else None
            datefmt = current_formatter.datefmt
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            datefmt = None
        
        handler.setFormatter(RedactingFormatter(fmt=fmt, datefmt=datefmt, config=config))


def redact_for_display(
    data: Union[str, Dict, pd.DataFrame],
    config: RedactionConfig = DEFAULT_CONFIG
) -> Union[str, Dict, pd.DataFrame]:
    """
    Universal redaction function that handles strings, dicts, or DataFrames.
    
    Args:
        data: Data to redact
        config: Redaction configuration
        
    Returns:
        Redacted data of the same type as input
    """
    if isinstance(data, str):
        return redact_string(data, config)
    elif isinstance(data, dict):
        return redact_dict(data, config)
    elif isinstance(data, pd.DataFrame):
        return redact_dataframe(data, config)
    else:
        return data


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def _test_redaction():
    """Test the redaction functionality."""
    if not PRESIDIO_AVAILABLE:
        print("ERROR: Presidio not installed. Cannot run tests.")
        print("Install with: pip install presidio-analyzer presidio-anonymizer")
        return
    
    test_cases = [
        "My email is john.doe@example.com and I live in New York",
        "Call me at 555-123-4567 or 1-800-555-0199",
        "SSN: 123-45-6789",
        "Credit card: 4532-1111-2222-3333",
        "My name is Sarah Johnson and I was born on 01/15/1990",
        "Server IP: 192.168.1.100",
        "Visit https://example.com/api?token=secret123",
        "Bitcoin wallet: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    ]
    
    print("=== Presidio Redaction Tests ===\n")
    config = RedactionConfig(show_entity_type=False)
    
    for test in test_cases:
        redacted = redact_string(test, config)
        print(f"Original: {test}")
        print(f"Redacted: {redacted}")
        print()
    
    # Test with entity type labels
    print("\n=== With Entity Type Labels ===\n")
    config_with_labels = RedactionConfig(show_entity_type=True)
    
    for test in test_cases[:3]:
        redacted = redact_string(test, config_with_labels)
        print(f"Original: {test}")
        print(f"Redacted: {redacted}")
        print()


if __name__ == "__main__":
    _test_redaction()
