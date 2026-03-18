# skill-diagram — Optimization Program

## Your Role

You are optimizing a Claude Code SKILL.md file that generates architecture diagrams as self-contained HTML. Your goal is to improve diagram quality across four binary criteria.

## Metric

Stochastic evaluation: 10 test prompts × 4 binary criteria × N samples. Score = mean criteria pass rate (higher is better). Current baseline: 1.8 / 4.0.

## Editable Files

- `examples/skill-diagram/SKILL.md`

## Evaluation Criteria

Each generated diagram is scored YES/NO on four questions:

1. **text_legibility** — Is ALL text clearly legible at 100% zoom?
2. **pastel_colors** — Does the diagram use ONLY pastel or muted colors?
3. **linear_layout** — Does the diagram follow a strictly linear layout with no crossing arrows?
4. **no_ordinals** — Does the diagram contain NO ordinal labels (1., 2., Step 1, Phase 2, etc.)?

## Optimization Strategies

1. **Explicit CSS font sizing** — Add minimum font-size rules to ensure legibility
2. **Color palette constraints** — Define specific pastel hex values in the skill instructions
3. **Layout directives** — Instruct the model to use top-to-bottom or left-to-right flow only
4. **Anti-patterns to prohibit** — Explicitly ban ordinal labels, crossing arrows, dark backgrounds
5. **Template structure** — Provide HTML/CSS scaffolding that enforces layout constraints
6. **Example-driven** — Include a concrete good/bad example showing what to avoid

## Constraints

- Only modify the file listed above
- The SKILL.md must remain a valid Claude Code skill definition
- Changes should improve criteria pass rates without regressing others

## Response Format

Before making edits, write:

## Hypothesis

[One sentence describing your optimization approach]

After edits, write:

## Tags

[Comma-separated categories: e.g., color-palette, layout-rules, font-sizing, anti-patterns]
