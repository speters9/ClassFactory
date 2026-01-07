# Archive Folder

This folder contains old versions of files that were replaced during the BeamerBot refactoring to support multiple output formats (LaTeX and PowerPoint).

## Archived Files

### `beamer_prompts.py` (Archived: January 6, 2026)
- **Replaced by:** `latex_prompts.py`
- **Reason:** Renamed to reflect format-specific nature. The new file includes backward compatibility aliases.
- **Content:** Original LaTeX Beamer prompt templates

### `beamer_slides.py` (Archived: January 6, 2026)
- **Replaced by:** `latex_slides.py`
- **Reason:** Renamed to reflect format-specific nature. The new file includes backward compatibility aliases.
- **Content:** Original Pydantic models for LaTeX Beamer slides (BeamerSlides, Slide classes)

### `BeamerBot_old_backup.py` (Archived: January 6, 2026)
- **Replaced by:** `BeamerBot.py` (factory pattern), `latex_slide_generator.py` (implementation)
- **Reason:** Refactored to use Factory pattern for multi-format support
- **Content:** Original monolithic BeamerBot class that only supported LaTeX output

## Migration Notes

All imports have been updated to use the new file names. The new files include backward compatibility aliases, so any external code using the old class names (e.g., `BeamerSlides`, `Slide`) will continue to work.

### New Architecture
```
BeamerBot (factory) → dispatches to:
    ├── LatexSlideGenerator (LaTeX/Beamer output)
    └── PptxSlideGenerator (PowerPoint output - coming soon)
```

Both generators inherit from `BaseSlideGenerator` which contains shared functionality.

## Safe to Delete?

Yes, these files can be safely deleted. They are kept only for reference purposes. All functionality has been migrated to the new architecture with backward compatibility maintained through aliases in:
- `latex_prompts.py` (exports `beamer_system_prompt` and `beamer_human_prompt`)
- `latex_slides.py` (exports `BeamerSlides` and `Slide`)
