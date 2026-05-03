"""Model package.

This package intentionally keeps ``__init__`` lightweight. Most call sites
import concrete submodules directly (for example ``model.model_manager`` or
``model.generation``), and avoiding eager re-exports here prevents circular
imports with the manual MultiLoRA runtime.
"""

__all__: list[str] = []
