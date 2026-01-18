"""License validation for HuggingFace datasets"""
from typing import Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)

# Licenses that are allowed for use in this AutoML platform
ALLOWED_LICENSES = {
    # Permissive open source
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "unlicense",
    
    # Creative Commons (excluding NC/ND variants that are too restrictive)
    "cc-by-4.0",
    "cc-by- sa-4.0",
    "cc0-1.0",
    "cc-by-nc-4.0",  # Non-commercial allowed for research/demo purposes
    
    # Other permissive licenses
    "isc",
    "zlib",
    "wtfpl",
}

# Keywords indicating restricted licenses
RESTRICTED_KEYWORDS = [
    "gpl",       # GNU General Public License (copyleft)
    "agpl",      # Affero GPL (copyleft + network)
    "proprietary",
    "commercial-only",
    "no-derivatives",
    "cc-by-nd",  # No derivatives
    "cc-by-nc-nd",  # No commercial + no derivatives
]


class LicenseValidator:
    """
    Validates HuggingFace dataset licenses for compliance.
    
    Ensures datasets can be legally used for training ML models and
    redistribution under the AutoML platform's terms.
    """
    
    @staticmethod
    def is_allowed(license_tag: Optional[str]) -> Tuple[bool, str]:
        """
        Check if a license allows use in this AutoML platform.
        
        Args:
            license_tag: License identifier from HF dataset metadata
            
        Returns:
            (is_valid, reason): Boolean validity and human-readable reason
            
        Examples:
            >>> LicenseValidator.is_allowed("mit")
            (True, "Permitted under MIT license")
            
            >>> LicenseValidator.is_allowed("gpl-3.0")
            (False, "Restricted license: GPL (copyleft)")
            
            >>> LicenseValidator.is_allowed(None)
            (False, "No license specified")
        """
        if not license_tag:
            return False, "No license specified"
        
        license_lower = license_tag.lower().strip()
        
        # Empty license
        if not license_lower:
            return False, "No license specified"
        
        # Check against allowed list (exact match)
        if license_lower in ALLOWED_LICENSES:
            logger.info(f"✓ License '{license_tag}' is allowed")
            return True, f"Permitted under {license_tag}"
        
        # Check for restricted keywords
        for keyword in RESTRICTED_KEYWORDS:
            if keyword in license_lower:
                logger.warning(f"✗ License '{license_tag}' contains restricted keyword '{keyword}'")
                return False, f"Restricted license: {license_tag} ({keyword})"
        
        # Check for specific patterns
        if license_lower.startswith("other"):
            return False, f"Custom/unknown license: {license_tag}"
        
        if license_lower in ["unknown", "unspecified", "none", "n/a"]:
            return False, "Unknown or unspecified license"
        
        # If we get here, it's a license we don't recognize
        # Be conservative - reject unless explicitly allowed
        logger.warning(f"? Unknown license '{license_tag}' - rejecting conservatively")
        return False, f"Unknown license: {license_tag} (not in allowed list)"
    
    @staticmethod
    def get_allowed_licenses() -> list[str]:
        """Return list of all allowed license identifiers"""
        return sorted(ALLOWED_LICENSES)
    
    @staticmethod
    def explain_rejection(license_tag: Optional[str]) -> str:
        """
        Provide a detailed explanation for why a license was rejected.
        
        Args:
            license_tag: The rejected license
            
        Returns:
            Human-readable explanation with suggestions
        """
        _, reason = LicenseValidator.is_allowed(license_tag)
        
        explanation = f"Dataset license '{license_tag}' was rejected: {reason}\n\n"
        
        if not license_tag:
            explanation += (
                "Datasets without a license cannot be used as their legal status is unclear. "
                "Look for datasets with explicit permissive licenses like MIT or Apache-2.0."
            )
        elif any(keyword in (license_tag or "").lower() for keyword in RESTRICTED_KEYWORDS):
            explanation += (
                "This license has restrictions that may conflict with model training and redistribution. "
                f"Allowed licenses include: {', '.join(LicenseValidator.get_allowed_licenses()[:5])} and others."
            )
        else:
            explanation += (
                f"This license is not in our approved list. "
                f"We support: {', '.join(LicenseValidator.get_allowed_licenses())}."
            )
        
        return explanation
