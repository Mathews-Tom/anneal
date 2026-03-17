# Diagram Generation Skill

Generate clear, visually appealing diagrams from technical concepts using HTML and inline SVG.

## Output Format

Return a single self-contained HTML file with inline CSS and SVG elements. No external dependencies.

## Style Guidelines

- Use soft, muted color palettes
- Keep layouts flowing in one direction (left-to-right or top-to-bottom)
- Make all text large enough to read without zooming
- Label components with descriptive names, not numbered steps
- Use arrows to show flow and relationships
- Group related elements visually with spacing or containers

## Structure

1. Parse the input concept into its key components and relationships
2. Determine the best flow direction based on the concept
3. Generate SVG elements for each component
4. Connect components with directional arrows
5. Apply consistent styling across all elements

## Constraints

- Maximum width: 800px
- Minimum font size: 14px
- Arrow heads must be visible
- Background should be white or very light
- No JavaScript required
