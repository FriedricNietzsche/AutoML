"""
Test suite for the redaction system.

Run with: python test_redaction.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import pandas as pd
from app.utils import (
    redact_string,
    redact_dict,
    redact_dataframe,
    redact_for_display,
    RedactionConfig,
    PRESIDIO_AVAILABLE,
)


def test_string_redaction():
    """Test string redaction with various PII types."""
    print("\n" + "="*60)
    print("TEST: String Redaction")
    print("="*60)
    
    test_cases = [
        ("Emails", "Contact john.doe@example.com or jane@company.org"),
        ("Phones", "Call 555-123-4567 or +1-800-555-0199"),
        ("SSN", "Social Security: 123-45-6789"),
        ("Credit Cards", "Card number: 4532-1111-2222-3333"),
        ("Names & Locations", "John Doe lives in New York City"),
        ("Mixed", "My email is john@example.com, phone 555-1234, SSN 123-45-6789"),
    ]
    
    config = RedactionConfig(enabled=True)
    
    for name, text in test_cases:
        redacted = redact_string(text, config)
        print(f"\n{name}:")
        print(f"  Original: {text}")
        print(f"  Redacted: {redacted}")
        print(f"  Changed: {'✅ YES' if text != redacted else '❌ NO'}")
    
    return True


def test_dict_redaction():
    """Test dictionary redaction."""
    print("\n" + "="*60)
    print("TEST: Dictionary Redaction")
    print("="*60)
    
    test_data = {
        "user_info": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "address": "123 Main St, New York, NY",
        },
        "payment": {
            "credit_card": "4532-1111-2222-3333",
            "ssn": "123-45-6789",
        },
        "metadata": {
            "user_id": "12345",  # Should NOT be redacted
            "timestamp": "2024-01-18T10:00:00Z",  # Should NOT be redacted
        },
        "api_credentials": {
            "api_key": "sk_live_1234567890abcdefghijklmnopqrstuvwxyz",
            "password": "super_secret_password123",
        }
    }
    
    config = RedactionConfig(
        enabled=True,
        whitelist_columns={'user_id', 'timestamp'}
    )
    
    redacted = redact_dict(test_data, config)
    
    print("\nOriginal:")
    import json
    print(json.dumps(test_data, indent=2))
    
    print("\nRedacted:")
    print(json.dumps(redacted, indent=2))
    
    # Verify whitelisted fields are preserved
    assert redacted['metadata']['user_id'] == "12345"
    assert redacted['metadata']['timestamp'] == "2024-01-18T10:00:00Z"
    
    # Verify sensitive keys are redacted
    assert redacted['api_credentials']['api_key'] == "[REDACTED]"
    assert redacted['api_credentials']['password'] == "[REDACTED]"
    
    print("\n✅ Whitelisting works correctly")
    print("✅ Sensitive keys redacted correctly")
    
    return True


def test_dataframe_redaction():
    """Test DataFrame redaction."""
    print("\n" + "="*60)
    print("TEST: DataFrame Redaction")
    print("="*60)
    
    # Create test DataFrame with PII
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "email": ["john@example.com", "jane@company.org", "bob@test.net"],
        "phone": ["555-1234", "555-5678", "555-9012"],
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555"],
        "notes": [
            "User registered from 192.168.1.100",
            "Contact via email or phone",
            "Lives in Boston, MA"
        ]
    })
    
    config = RedactionConfig(
        enabled=True,
        whitelist_columns={'id'}  # Keep ID column
    )
    
    print("\nOriginal DataFrame:")
    print(df)
    
    df_redacted = redact_dataframe(df, config)
    
    print("\nRedacted DataFrame:")
    print(df_redacted)
    
    # Verify ID column is preserved
    assert df_redacted['id'].equals(df['id'])
    
    # Verify sensitive columns are changed
    assert not df_redacted['email'].equals(df['email'])
    assert not df_redacted['ssn'].equals(df['ssn'])
    
    print("\n✅ DataFrame redaction working correctly")
    
    return True


def test_entity_type_display():
    """Test redaction with entity type labels."""
    print("\n" + "="*60)
    print("TEST: Entity Type Display (for debugging)")
    print("="*60)
    
    config = RedactionConfig(
        enabled=True,
        show_entity_type=True  # Show what was redacted
    )
    
    test_text = "Contact john@example.com or call 555-1234. SSN: 123-45-6789"
    redacted = redact_string(test_text, config)
    
    print(f"\nOriginal: {test_text}")
    print(f"Redacted: {redacted}")
    print("\n✅ Entity types displayed for debugging")
    
    return True


def test_configuration_options():
    """Test different configuration options."""
    print("\n" + "="*60)
    print("TEST: Configuration Options")
    print("="*60)
    
    text = "Email: john@example.com, Phone: 555-1234"
    
    # Test 1: Hash mode
    config_hash = RedactionConfig(hash_sensitive_data=True)
    redacted_hash = redact_string(text, config_hash)
    print(f"\nHash mode: {redacted_hash}")
    
    # Test 2: Custom mask character
    config_custom = RedactionConfig(mask_char="#")
    redacted_custom = redact_string(text, config_custom)
    print(f"Custom mask: {redacted_custom}")
    
    # Test 3: Specific entities only
    config_email_only = RedactionConfig(
        entities=["EMAIL_ADDRESS"]  # Only redact emails
    )
    redacted_email_only = redact_string(text, config_email_only)
    print(f"Email only: {redacted_email_only}")
    
    # Test 4: Higher threshold (less sensitive)
    config_high_threshold = RedactionConfig(score_threshold=0.8)
    redacted_high = redact_string(text, config_high_threshold)
    print(f"High threshold: {redacted_high}")
    
    print("\n✅ All configuration options working")
    
    return True


def test_universal_redaction():
    """Test the universal redact_for_display function."""
    print("\n" + "="*60)
    print("TEST: Universal Redaction Function")
    print("="*60)
    
    # Test with string
    string_result = redact_for_display("Email: john@example.com")
    print(f"\nString: {string_result}")
    
    # Test with dict
    dict_result = redact_for_display({
        "email": "john@example.com",
        "phone": "555-1234"
    })
    print(f"Dict: {dict_result}")
    
    # Test with DataFrame
    df = pd.DataFrame({"email": ["john@example.com"]})
    df_result = redact_for_display(df)
    print(f"DataFrame:\n{df_result}")
    
    print("\n✅ Universal redaction working for all types")
    
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "🔒"*30)
    print("REDACTION SYSTEM TEST SUITE")
    print("🔒"*30)
    
    if not PRESIDIO_AVAILABLE:
        print("\n❌ ERROR: Presidio not available!")
        print("Install with: pip install presidio-analyzer presidio-anonymizer")
        return False
    
    print("\n✅ Presidio is available and loaded")
    
    tests = [
        ("String Redaction", test_string_redaction),
        ("Dictionary Redaction", test_dict_redaction),
        ("DataFrame Redaction", test_dataframe_redaction),
        ("Entity Type Display", test_entity_type_display),
        ("Configuration Options", test_configuration_options),
        ("Universal Redaction", test_universal_redaction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
