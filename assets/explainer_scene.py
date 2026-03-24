"""Anneal explainer video — 5 scenes, ~75 seconds total.

Before/after narrative using the code-golf example.
Shows the concrete workflow: pain → setup → loop → result.

Render: python3 scripts/render_video.py explainer_scene.py <SceneName> --quality high --format mp4
Concat: ffmpeg -f concat -safe 0 -i concat_list.txt -c copy explainer.mp4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import *

# === Palette ===
BG = "#0F172A"
INDIGO = "#4F46E5"
INDIGO_LIGHT = "#818CF8"
AMBER = "#F59E0B"
AMBER_LIGHT = "#FBBF24"
MINT = "#34D399"
CORAL = "#F87171"
WHITE = "#F8FAFC"
SLATE = "#94A3B8"
SLATE_DARK = "#475569"
TERM_GREEN = "#4ADE80"


def _terminal_box(width: float = 10, height: float = 5.5) -> VGroup:
    """Reusable terminal window chrome."""
    bg = RoundedRectangle(
        width=width, height=height, corner_radius=0.2,
        color=SLATE_DARK, fill_opacity=0.15, stroke_width=1.5,
    )
    # Traffic light dots
    dots = VGroup(
        Dot(radius=0.06, color="#FF5F57"),
        Dot(radius=0.06, color="#FFBD2E"),
        Dot(radius=0.06, color="#28CA41"),
    ).arrange(RIGHT, buff=0.12)
    dots.move_to(bg.get_corner(UL) + RIGHT * 0.5 + DOWN * 0.25)
    return VGroup(bg, dots)


# ============================================================
# Scene 1: The Pain (0–12s)
# ============================================================
class Scene1Pain(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Developer tweaking a file
        title = Text("Manual optimization", font_size=36, weight=BOLD, color=SLATE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        # Score display
        score_label = Text("coverage", font_size=20, color=SLATE)
        score_label.move_to(UP * 1.5)
        score = Text("45%", font_size=80, color=INDIGO_LIGHT, font="Courier New")
        score.next_to(score_label, DOWN, buff=0.2)
        self.play(FadeIn(score_label), Write(score), run_time=0.8)
        self.wait(0.3)

        # Manual attempts — tedious incremental changes
        attempts = [
            ("add test", "+3%", "48%", MINT),
            ("fix edge case", "+2%", "50%", MINT),
            ("refactor", "-1%", "49%", CORAL),
            ("add mock", "+1%", "50%", MINT),
            ("more tests", "+2%", "52%", MINT),
            ("missed path", "+0%", "52%", CORAL),
        ]

        attempt_label = Text("", font_size=18, color=SLATE)
        attempt_label.move_to(DOWN * 1.2)

        for desc, delta_text, new_val, color in attempts:
            # Show what was tried
            new_attempt = Text(desc, font_size=18, color=SLATE)
            new_attempt.move_to(DOWN * 1.2)
            delta = Text(delta_text, font_size=24, color=color)
            delta.next_to(score, RIGHT, buff=0.6)

            self.play(
                FadeIn(delta, shift=UP * 0.15),
                ReplacementTransform(attempt_label, new_attempt),
                run_time=0.25,
            )
            attempt_label = new_attempt

            new_score = Text(new_val, font_size=80, color=INDIGO_LIGHT,
                             font="Courier New")
            new_score.move_to(score)
            self.play(
                ReplacementTransform(score, new_score),
                FadeOut(delta),
                run_time=0.3,
            )
            score = new_score

        self.wait(0.4)

        # Time annotation
        time_spent = Text("2 hours later...", font_size=22, color=AMBER_LIGHT,
                          slant=ITALIC)
        time_spent.move_to(DOWN * 2.2)
        self.play(FadeIn(time_spent, shift=UP * 0.15), run_time=0.5)
        self.wait(1.2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.4)
        self.wait(0.2)


# ============================================================
# Scene 2: Three Commands (12–25s)
# ============================================================
class Scene2Setup(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        title = Text("With anneal", font_size=36, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.4)

        term = _terminal_box(width=11, height=5)
        term.move_to(DOWN * 0.3)
        self.play(FadeIn(term), run_time=0.3)

        # Commands typed into terminal
        commands = [
            ("$ anneal init", TERM_GREEN, 0.6),
            ("  Initialized .anneal/ in current repo", SLATE, 0.3),
            ("", None, 0.2),
            ("$ anneal register \\", TERM_GREEN, 0.5),
            ("    --name test-coverage \\", WHITE, 0.15),
            ("    --artifact tests/test_calc.py \\", WHITE, 0.15),
            ("    --eval-mode deterministic \\", WHITE, 0.15),
            ("    --run-cmd \"pytest --cov\" \\", WHITE, 0.15),
            ("    --direction maximize", WHITE, 0.15),
            ("  Registered target: test-coverage", SLATE, 0.3),
            ("", None, 0.2),
            ("$ anneal run --target test-coverage --experiments 20", TERM_GREEN, 0.6),
        ]

        y_pos = term[0].get_top()[1] - 0.7
        x_pos = term[0].get_left()[0] + 0.5
        cmd_mobs = []

        for text, color, wait_time in commands:
            if not text:
                y_pos -= 0.25
                self.wait(wait_time)
                continue
            cmd = Text(text, font_size=15, color=color, font="Courier New")
            cmd.move_to(np.array([x_pos, y_pos, 0]))
            cmd.align_to(term[0].get_left() + RIGHT * 0.4, LEFT)
            cmd_mobs.append(cmd)
            self.play(FadeIn(cmd, shift=RIGHT * 0.1), run_time=wait_time)
            y_pos -= 0.3

        self.wait(0.8)

        # Highlight: "That's it. Walk away."
        hint = Text("Walk away. Come back to results.",
                     font_size=24, color=AMBER_LIGHT, weight=BOLD)
        hint.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(hint, shift=UP * 0.2), run_time=0.5)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.4)
        self.wait(0.2)


# ============================================================
# Scene 3: The Loop Running (25–50s)
# ============================================================
class Scene3Loop(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Split screen: terminal left, score graph right
        term = _terminal_box(width=7, height=5.5)
        term.move_to(LEFT * 3.3 + DOWN * 0.1)
        self.play(FadeIn(term), run_time=0.3)

        # Score chart on the right
        chart_title = Text("coverage %", font_size=18, color=SLATE)
        chart_title.move_to(RIGHT * 3.5 + UP * 2.5)
        self.play(FadeIn(chart_title), run_time=0.2)

        # Axes for the chart
        ax_origin = np.array([1.5, -2.5, 0])
        ax_width = 4.5
        ax_height = 4.5
        x_axis = Line(ax_origin, ax_origin + RIGHT * ax_width,
                       color=SLATE_DARK, stroke_width=1)
        y_axis = Line(ax_origin, ax_origin + UP * ax_height,
                       color=SLATE_DARK, stroke_width=1)
        self.play(Create(x_axis), Create(y_axis), run_time=0.3)

        # Y-axis labels
        for pct, y_frac in [(40, 0), (60, 0.33), (80, 0.67), (95, 0.92)]:
            label = Text(f"{pct}", font_size=12, color=SLATE, font="Courier New")
            label.move_to(ax_origin + UP * (y_frac * ax_height) + LEFT * 0.35)
            self.add(label)

        # Experiments data (test coverage climbing)
        experiments = [
            (1, "KEPT", 52, MINT),
            (2, "KEPT", 61, MINT),
            (3, "DISCARDED", 58, CORAL),
            (4, "KEPT", 68, MINT),
            (5, "DISCARDED", 65, CORAL),
            (6, "KEPT", 74, MINT),
            (7, "KEPT", 79, MINT),
            (8, "DISCARDED", 76, CORAL),
            (9, "KEPT", 83, MINT),
            (10, "KEPT", 86, MINT),
            (11, "DISCARDED", 84, CORAL),
            (12, "KEPT", 89, MINT),
            (13, "KEPT", 91, MINT),
            (14, "DISCARDED", 90, CORAL),
            (15, "KEPT", 93, MINT),
            (16, "DISCARDED", 92, CORAL),
            (17, "KEPT", 95, MINT),
        ]

        y_start = term[0].get_top()[1] - 0.6
        x_left = term[0].get_left()[0] + 0.3
        chart_dots = []
        prev_best = 45

        for exp_num, outcome, score, color in experiments:
            # Terminal line
            if outcome == "KEPT":
                prev_best = score
            display_score = prev_best if outcome == "DISCARDED" else score

            line_text = f"exp {exp_num:>2}  {outcome:<10} {score}%"
            line = Text(line_text, font_size=13, color=color, font="Courier New")
            y_pos = y_start - ((exp_num - 1) % 10) * 0.35
            if exp_num == 11:
                # Clear terminal for second batch
                term_lines = [m for m in self.mobjects
                              if isinstance(m, Text) and hasattr(m, 'font')
                              and m.font == "Courier New"
                              and m.get_center()[0] < 0]
                self.play(*[FadeOut(t) for t in term_lines], run_time=0.15)
                y_pos = y_start

            line.move_to(np.array([x_left, y_pos, 0]))
            line.align_to(term[0].get_left() + RIGHT * 0.3, LEFT)

            # Chart dot
            x_frac = (exp_num - 1) / (len(experiments) - 1)
            y_frac = (display_score - 40) / (100 - 40)
            dot_pos = ax_origin + RIGHT * (x_frac * ax_width) + UP * (y_frac * ax_height)
            dot = Dot(dot_pos, radius=0.07, color=color)

            # Line connecting to previous dot
            anims = [FadeIn(line, shift=RIGHT * 0.05), FadeIn(dot, scale=0.5)]
            if chart_dots:
                conn = Line(chart_dots[-1].get_center(), dot.get_center(),
                            color=SLATE_DARK, stroke_width=1, stroke_opacity=0.4)
                anims.append(Create(conn))
            chart_dots.append(dot)

            self.play(*anims, run_time=0.35)

        # Final score highlight
        final_score = Text("95%", font_size=48, color=MINT, font="Courier New",
                           weight=BOLD)
        final_score.move_to(RIGHT * 3.5 + DOWN * 1.8)
        final_label = Text("final coverage", font_size=16, color=SLATE)
        final_label.next_to(final_score, DOWN, buff=0.15)
        self.play(Write(final_score), FadeIn(final_label), run_time=0.6)
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.2)


# ============================================================
# Scene 4: Before / After (50–65s)
# ============================================================
class Scene4BeforeAfter(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # === BEFORE side ===
        before_title = Text("Before", font_size=28, color=SLATE, weight=BOLD)
        before_title.move_to(LEFT * 3.5 + UP * 3)

        before_score = Text("45%", font_size=64, color=SLATE, font="Courier New")
        before_score.move_to(LEFT * 3.5 + UP * 1.8)

        before_detail = Text("3 basic tests\nno edge cases\nno error paths",
                             font_size=16, color=SLATE, font="Courier New",
                             line_spacing=1.4)
        before_detail.move_to(LEFT * 3.5 + UP * 0.3)

        # === AFTER side ===
        after_title = Text("After", font_size=28, color=MINT, weight=BOLD)
        after_title.move_to(RIGHT * 3.5 + UP * 3)

        after_score = Text("95%", font_size=64, color=MINT, font="Courier New")
        after_score.move_to(RIGHT * 3.5 + UP * 1.8)

        after_detail = Text("21 tests\nboundary checks\nerror paths covered",
                            font_size=16, color=MINT, font="Courier New",
                            line_spacing=1.4)
        after_detail.move_to(RIGHT * 3.5 + UP * 0.3)

        # Arrow
        arrow = Arrow(LEFT * 1.2, RIGHT * 1.2, color=AMBER, stroke_width=3,
                      max_tip_length_to_length_ratio=0.15)
        arrow.move_to(UP * 1.8)

        # Divider
        divider = DashedLine(UP * 3.3, DOWN * 2.5, color=SLATE_DARK,
                              stroke_width=1, dash_length=0.15)

        # Animate before
        self.play(
            FadeIn(before_title), Write(before_score),
            FadeIn(before_detail, shift=UP * 0.1),
            run_time=0.8,
        )
        self.wait(0.5)

        # Divider and arrow
        self.play(Create(divider), GrowArrow(arrow), run_time=0.5)

        # Animate after
        self.play(
            FadeIn(after_title), Write(after_score),
            FadeIn(after_detail, shift=UP * 0.1),
            run_time=0.8,
        )
        self.wait(0.8)

        # Stats bar at bottom
        stats = VGroup(
            Text("17 experiments", font_size=22, color=WHITE),
            Text("$2.40 total cost", font_size=22, color=AMBER_LIGHT),
            Text("~25 minutes", font_size=22, color=SLATE),
        ).arrange(RIGHT, buff=1.5)
        stats.move_to(DOWN * 1.5)
        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in stats], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.5)

        # Other use cases flash
        other_title = Text("Works on anything measurable", font_size=22,
                           color=AMBER_LIGHT, weight=BOLD)
        other_title.move_to(DOWN * 2.5)
        self.play(FadeIn(other_title, shift=UP * 0.15), run_time=0.5)

        use_cases = VGroup(
            Text("prompts", font_size=18, color=INDIGO_LIGHT),
            Text("API latency", font_size=18, color=INDIGO_LIGHT),
            Text("bundle size", font_size=18, color=INDIGO_LIGHT),
            Text("build configs", font_size=18, color=INDIGO_LIGHT),
        ).arrange(RIGHT, buff=1.0)
        use_cases.move_to(DOWN * 3.1)
        self.play(
            LaggedStart(*[FadeIn(u, shift=UP * 0.1) for u in use_cases], lag_ratio=0.15),
            run_time=0.6,
        )
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 5: End Card (65–75s)
# ============================================================
class Scene5EndCard(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Logo
        logo = Text("anneal", font_size=72, weight=BOLD, color=WHITE)
        logo.move_to(UP * 1)
        self.play(FadeIn(logo, scale=0.85), run_time=0.8)

        # Tagline
        tagline = Text(
            "Let an AI agent improve your code — overnight, unattended.",
            font_size=22, color=SLATE,
        )
        tagline.next_to(logo, DOWN, buff=0.5)
        self.play(FadeIn(tagline, shift=UP * 0.15), run_time=0.5)
        self.wait(0.5)

        # Install command
        install_bg = RoundedRectangle(
            width=6.5, height=0.8, corner_radius=0.15,
            color=SLATE_DARK, fill_opacity=0.2, stroke_width=1,
        )
        install_bg.move_to(DOWN * 1)
        install_cmd = Text("uv tool install anneal-cli", font_size=22,
                           color=TERM_GREEN, font="Courier New")
        install_cmd.move_to(install_bg)
        self.play(FadeIn(install_bg), Write(install_cmd), run_time=0.6)

        # GitHub link with icon
        gh_icon = SVGMobject(
            str(Path(__file__).parent / "github-mark.svg"),
            height=0.35,
        ).set_color(SLATE)
        gh_text = Text("github.com/Mathews-Tom/anneal", font_size=16, color=SLATE)
        gh_row = VGroup(gh_icon, gh_text).arrange(RIGHT, buff=0.2)
        gh_row.move_to(DOWN * 2)
        self.play(FadeIn(gh_row), run_time=0.4)

        self.wait(4)
