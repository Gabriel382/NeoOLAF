"""Document profile support for NeoOLAF.

Profiles keep document-specific assumptions outside the generic NeoOLAF
layers.  A profile can select chunking strategies, prompts, relation mappings,
RAG spaces, and evaluation schema details without hard-coding those details in
core code.
"""

from neoolaf.profiles.document_profile import DocumentProfile
from neoolaf.profiles.profile_loader import load_document_profile

__all__ = ["DocumentProfile", "load_document_profile"]
