"""Anneal explainer video — 6 scenes, ~90 seconds total.

Render: python3 scripts/render_video.py explainer_scene.py <SceneName> --quality high --format mp4
Concat: ffmpeg -f concat -safe 0 -i concat_list.txt -c copy explainer.mp4
"""
from __future__ import annotations

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


# ============================================================
# Scene 1: The Problem (0–15s)
# ============================================================
class Scene1Problem(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Title
        title = Text("The Problem", font_size=42, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.6)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.6)

        # Metric gauge — large percentage
        score = Text("72%", font_size=96, color=INDIGO_LIGHT,
                      font="Courier New")
        score.move_to(ORIGIN + UP * 0.3)
        label = Text("optimization score", font_size=22, color=SLATE)
        label.next_to(score, DOWN, buff=0.3)
        self.play(Write(score), FadeIn(label), run_time=1)
        self.wait(0.5)

        # Manual increments — tedious changes
        changes = [
            ("+1%", "73%", MINT),
            ("-1%", "72%", CORAL),
            ("+0.5%", "72.5%", MINT),
            ("-0.3%", "72.2%", CORAL),
            ("+0.8%", "73%", MINT),
        ]
        for delta_text, new_val, color in changes:
            delta = Text(delta_text, font_size=28, color=color)
            delta.next_to(score, RIGHT, buff=0.8)
            self.play(FadeIn(delta, shift=UP * 0.2), run_time=0.25)
            new_score = Text(new_val, font_size=96, color=INDIGO_LIGHT,
                             font="Courier New")
            new_score.move_to(score)
            self.play(
                ReplacementTransform(score, new_score),
                FadeOut(delta),
                run_time=0.35,
            )
            score = new_score

        self.wait(0.3)

        # Punchline
        msg = Text("Manual optimization doesn't scale.",
                    font_size=30, color=AMBER_LIGHT, weight=BOLD)
        msg.next_to(label, DOWN, buff=0.8)
        self.play(FadeIn(msg, shift=UP * 0.2), run_time=0.7)
        self.wait(1.5)

        # Transition out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 2: The Loop (15–35s)
# ============================================================
class Scene2Loop(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Section title
        title = Text("The Core Loop", font_size=42, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Three nodes arranged in a triangle
        radius = 2.2
        positions = [
            radius * np.array([np.cos(PI / 2), np.sin(PI / 2), 0]),         # top
            radius * np.array([np.cos(PI / 2 - 2 * PI / 3), np.sin(PI / 2 - 2 * PI / 3), 0]),  # bottom-right
            radius * np.array([np.cos(PI / 2 + 2 * PI / 3), np.sin(PI / 2 + 2 * PI / 3), 0]),  # bottom-left
        ]
        center_offset = DOWN * 0.3

        names = ["Artifact", "Eval", "Agent"]
        colors = [INDIGO_LIGHT, MINT, AMBER]
        icons = []

        # Artifact — document shape
        doc = VGroup(
            RoundedRectangle(width=1.4, height=1.7, corner_radius=0.1,
                             color=INDIGO_LIGHT, fill_opacity=0.15, stroke_width=2),
            # Lines inside document
            Line(LEFT * 0.4, RIGHT * 0.4, color=INDIGO_LIGHT, stroke_width=1.5).shift(UP * 0.3),
            Line(LEFT * 0.4, RIGHT * 0.3, color=INDIGO_LIGHT, stroke_width=1.5),
            Line(LEFT * 0.4, RIGHT * 0.2, color=INDIGO_LIGHT, stroke_width=1.5).shift(DOWN * 0.3),
        )
        doc.move_to(positions[0] + center_offset)

        # Eval — gauge
        gauge = VGroup(
            Circle(radius=0.7, color=MINT, stroke_width=2, fill_opacity=0.1),
            Line(ORIGIN, UP * 0.5 + RIGHT * 0.2, color=MINT, stroke_width=3),
        )
        gauge.move_to(positions[1] + center_offset)

        # Agent — brain circle
        brain = VGroup(
            Circle(radius=0.7, color=AMBER, stroke_width=2, fill_opacity=0.1),
            Text("AI", font_size=24, color=AMBER, weight=BOLD),
        )
        brain.move_to(positions[2] + center_offset)

        nodes = [doc, gauge, brain]

        # Labels below each node
        labels = []
        for i, name in enumerate(names):
            lbl = Text(name, font_size=22, color=colors[i])
            lbl.next_to(nodes[i], DOWN, buff=0.25)
            labels.append(lbl)

        # Animate nodes appearing
        self.play(
            LaggedStart(*[FadeIn(n, scale=0.7) for n in nodes], lag_ratio=0.2),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.15),
            run_time=0.6,
        )

        # Arrows between nodes
        arrows = []
        action_labels = ["mutate", "score", "learn"]
        action_colors = [INDIGO_LIGHT, MINT, AMBER]
        for i in range(3):
            start = nodes[i].get_center()
            end = nodes[(i + 1) % 3].get_center()
            arrow = Arrow(start, end, buff=0.95, color=SLATE_DARK, stroke_width=2.5,
                          max_tip_length_to_length_ratio=0.12)
            arrows.append(arrow)
            # Action label on the arrow
            mid = (start + end) / 2
            direction = end - start
            perp = np.array([-direction[1], direction[0], 0])
            perp = perp / (np.linalg.norm(perp) + 1e-8) * 0.35
            al = Text(action_labels[i], font_size=16, color=action_colors[i])
            al.move_to(mid + perp)

        for arrow in arrows:
            self.play(GrowArrow(arrow), run_time=0.3)

        self.wait(0.3)

        # Animate cycle pulse — highlight travels around
        for _cycle in range(3):
            for i in range(3):
                self.play(
                    nodes[i][0].animate.set_fill(colors[i], opacity=0.5),
                    run_time=0.2,
                )
                self.play(
                    nodes[i][0].animate.set_fill(colors[i], opacity=0.15),
                    run_time=0.15,
                )

        # Tagline
        tagline = Text("Define what to improve.\nThe loop runs itself.",
                        font_size=26, color=WHITE, line_spacing=1.3)
        tagline.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.7)
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 3: Two Eval Modes (35–50s)
# ============================================================
class Scene3EvalModes(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        title = Text("Two Evaluation Modes", font_size=42, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Divider
        divider = Line(UP * 2.5, DOWN * 2.5, color=SLATE_DARK, stroke_width=1)
        self.play(Create(divider), run_time=0.3)

        # === LEFT: Deterministic ===
        left_title = Text("Deterministic", font_size=28, color=INDIGO_LIGHT, weight=BOLD)
        left_title.move_to(LEFT * 3.5 + UP * 2)
        self.play(Write(left_title), run_time=0.4)

        # Terminal box
        term_bg = RoundedRectangle(width=5.5, height=3, corner_radius=0.15,
                                    color=SLATE_DARK, fill_opacity=0.15, stroke_width=1)
        term_bg.move_to(LEFT * 3.5 + DOWN * 0.3)
        self.play(FadeIn(term_bg), run_time=0.3)

        # Command
        cmd = Text("$ pytest --cov | parse", font_size=18, color=SLATE,
                    font="Courier New")
        cmd.move_to(term_bg.get_top() + DOWN * 0.6)
        self.play(Write(cmd), run_time=0.5)

        # Scores appearing
        scores = ["72.3", "74.1", "76.8"]
        score_colors = [SLATE, AMBER_LIGHT, MINT]
        prev_score_mob = None
        for i, (s, c) in enumerate(zip(scores, score_colors)):
            score_mob = Text(s, font_size=48, color=c, font="Courier New")
            score_mob.move_to(term_bg.get_center() + DOWN * 0.2)
            if prev_score_mob:
                self.play(ReplacementTransform(prev_score_mob, score_mob), run_time=0.5)
            else:
                self.play(Write(score_mob), run_time=0.5)
            prev_score_mob = score_mob
            self.wait(0.3)

        # === RIGHT: Stochastic ===
        right_title = Text("Stochastic", font_size=28, color=AMBER, weight=BOLD)
        right_title.move_to(RIGHT * 3.5 + UP * 2)
        self.play(Write(right_title), run_time=0.4)

        stoch_bg = RoundedRectangle(width=5.5, height=3, corner_radius=0.15,
                                     color=SLATE_DARK, fill_opacity=0.15, stroke_width=1)
        stoch_bg.move_to(RIGHT * 3.5 + DOWN * 0.3)
        self.play(FadeIn(stoch_bg), run_time=0.3)

        # Criteria with votes
        criteria = [
            ("Scannable?", ["YES", "YES", "NO"], True),
            ("Cited?", ["NO", "YES", "YES"], True),
            ("Clear?", ["YES", "YES", "YES"], True),
        ]
        y_offset = -0.8
        for crit_name, votes, result in criteria:
            y_pos = stoch_bg.get_top()[1] + y_offset
            crit_text = Text(crit_name, font_size=18, color=WHITE)
            crit_text.move_to(RIGHT * 2.3 + UP * y_pos)

            vote_group = VGroup()
            for j, v in enumerate(votes):
                vc = MINT if v == "YES" else CORAL
                vm = Text(v, font_size=14, color=vc, font="Courier New")
                vm.move_to(RIGHT * (3.6 + j * 0.7) + UP * y_pos)
                vote_group.add(vm)

            self.play(FadeIn(crit_text), run_time=0.2)
            self.play(LaggedStart(*[FadeIn(v, scale=0.8) for v in vote_group],
                                   lag_ratio=0.1), run_time=0.4)
            y_offset -= 0.7

        # Confidence interval bar
        ci_label = Text("CI: [0.71, 0.89]", font_size=16, color=SLATE,
                         font="Courier New")
        ci_label.move_to(stoch_bg.get_bottom() + UP * 0.3)
        self.play(FadeIn(ci_label), run_time=0.3)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 4: Search Strategies (50–65s)
# ============================================================
class Scene4Search(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        title = Text("Search Strategies", font_size=42, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Optimization landscape
        x_range = np.linspace(-6, 6, 300)
        landscape_y = (
            -0.6 * np.sin(1.2 * x_range)
            - 0.4 * np.sin(2.5 * x_range + 1)
            - 0.3 * np.cos(0.8 * x_range + 2)
            + 0.1 * x_range
        )
        # Normalize to fit canvas
        landscape_y = (landscape_y - landscape_y.min()) / (landscape_y.max() - landscape_y.min())
        landscape_y = landscape_y * 3 - 2.5  # map to y ∈ [-2.5, 0.5]

        # Draw landscape curve
        points = [np.array([x_range[i], landscape_y[i], 0]) for i in range(len(x_range))]
        landscape = VMobject(color=INDIGO_LIGHT, stroke_width=2, stroke_opacity=0.6)
        landscape.set_points_smoothly(points)
        self.play(Create(landscape), run_time=1)

        # Find local minima for path visualization
        # Local min 1 around x=-3.5, Local min 2 around x=1, Global min around x=4
        def y_at(x):
            idx = int((x + 6) / 12 * 299)
            idx = max(0, min(298, idx))
            return landscape_y[idx]

        # === Greedy path (green) — gets stuck in first valley ===
        greedy_xs = [-5.5, -4.5, -3.5, -3.0]
        greedy_points = [Dot(np.array([x, y_at(x), 0]), color=MINT, radius=0.08)
                         for x in greedy_xs]
        greedy_label = Text("Greedy", font_size=18, color=MINT)
        greedy_label.next_to(greedy_points[-1], UP, buff=0.2)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in greedy_points],
                               lag_ratio=0.2), run_time=1)
        # Stuck indicator
        stuck = Text("stuck", font_size=14, color=CORAL)
        stuck.next_to(greedy_points[-1], DOWN, buff=0.15)
        self.play(FadeIn(greedy_label), FadeIn(stuck), run_time=0.4)
        self.wait(0.5)

        # === Annealing path (amber) — escapes and finds deeper valley ===
        anneal_xs = [-5.5, -4.5, -3.5, -2.5, -1.5, 0.0, 1.0, 1.5]
        anneal_dots = []
        for i, x in enumerate(anneal_xs):
            c = AMBER_LIGHT if y_at(x) > y_at(anneal_xs[max(0, i - 1)]) else AMBER
            anneal_dots.append(Dot(np.array([x, y_at(x), 0]), color=c, radius=0.08))

        anneal_label = Text("Annealing", font_size=18, color=AMBER)
        anneal_label.next_to(anneal_dots[-1], UP, buff=0.2)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in anneal_dots],
                               lag_ratio=0.15), run_time=1.2)
        self.play(FadeIn(anneal_label), run_time=0.3)

        # Uphill annotation
        uphill = Text("escapes", font_size=14, color=AMBER_LIGHT)
        uphill.move_to(np.array([-2.0, y_at(-2.0) + 0.4, 0]))
        self.play(FadeIn(uphill), run_time=0.3)
        self.wait(0.5)

        # === Pareto path (blue) — fans into multiple valleys ===
        pareto_xs_a = [-5.5, -3.5]
        pareto_xs_b = [-5.5, -1.0, 1.0]
        pareto_xs_c = [-5.5, 0.5, 3.5, 4.5]

        for path_xs in [pareto_xs_a, pareto_xs_b, pareto_xs_c]:
            pdots = [Dot(np.array([x, y_at(x), 0]), color=BLUE, radius=0.06)
                     for x in path_xs]
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in pdots],
                                   lag_ratio=0.1), run_time=0.5)

        pareto_label = Text("Pareto", font_size=18, color=BLUE)
        pareto_label.move_to(np.array([4.5, y_at(4.5) + 0.4, 0]))
        trade_text = Text("trade-offs", font_size=14, color=BLUE_B)
        trade_text.next_to(pareto_label, DOWN, buff=0.15)
        self.play(FadeIn(pareto_label), FadeIn(trade_text), run_time=0.3)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 5: Knowledge Compounding (65–80s)
# ============================================================
class Scene5Knowledge(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        title = Text("Knowledge Compounding", font_size=42, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Timeline axis
        axis = Line(LEFT * 6, RIGHT * 2, color=SLATE_DARK, stroke_width=1.5)
        axis.move_to(DOWN * 0.5)
        axis_label = Text("experiments", font_size=16, color=SLATE)
        axis_label.next_to(axis, DOWN, buff=0.15)
        self.play(Create(axis), FadeIn(axis_label), run_time=0.5)

        # Experiment dots appearing on timeline
        n_dots = 20
        dots = []
        for i in range(n_dots):
            x = -5.5 + i * 0.5
            # Early experiments: more failures. Later: more successes
            success_prob = 0.3 + 0.5 * (i / n_dots)
            is_kept = np.random.RandomState(seed=i + 42).random() < success_prob
            color = MINT if is_kept else CORAL
            dot = Dot(np.array([x, -0.5, 0]), color=color, radius=0.1)
            dots.append(dot)

        # Animate in groups
        for batch_start in range(0, n_dots, 4):
            batch = dots[batch_start:batch_start + 4]
            self.play(
                LaggedStart(*[FadeIn(d, scale=0.5) for d in batch], lag_ratio=0.08),
                run_time=0.5,
            )

        # Learnings panel (right side)
        panel_bg = RoundedRectangle(width=4.5, height=3.5, corner_radius=0.15,
                                     color=SLATE_DARK, fill_opacity=0.1, stroke_width=1)
        panel_bg.move_to(RIGHT * 4 + UP * 1.5)
        panel_title = Text("Learnings", font_size=22, color=WHITE, weight=BOLD)
        panel_title.move_to(panel_bg.get_top() + DOWN * 0.3)
        self.play(FadeIn(panel_bg), FadeIn(panel_title), run_time=0.4)

        # Learning entries appearing
        learnings = [
            ("+0.15", "clarity", MINT),
            ("-0.08", "accuracy", CORAL),
            ("+0.12", "tone", MINT),
            ("+0.20", "brevity", MINT),
        ]
        for i, (delta, name, color) in enumerate(learnings):
            entry = Text(f"{delta} {name}", font_size=18, color=color,
                          font="Courier New")
            entry.move_to(panel_bg.get_top() + DOWN * (0.7 + i * 0.45) + LEFT * 0.3)
            entry.align_to(panel_bg.get_left() + RIGHT * 0.4, LEFT)
            self.play(FadeIn(entry, shift=RIGHT * 0.2), run_time=0.35)

        # Success rate increasing
        rate_label = Text("success rate", font_size=16, color=SLATE)
        rate_label.move_to(LEFT * 3.5 + DOWN * 2)

        rates = ["30%", "45%", "62%", "78%"]
        rate_colors = [CORAL, AMBER_LIGHT, AMBER, MINT]
        prev_rate = None
        for r, c in zip(rates, rate_colors):
            rate_mob = Text(r, font_size=36, color=c, font="Courier New")
            rate_mob.next_to(rate_label, RIGHT, buff=0.4)
            if prev_rate:
                self.play(ReplacementTransform(prev_rate, rate_mob), run_time=0.4)
            else:
                self.play(Write(rate_mob), run_time=0.4)
            prev_rate = rate_mob
            self.wait(0.2)

        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        self.wait(0.3)


# ============================================================
# Scene 6: The Result (80–90s)
# ============================================================
class Scene6Result(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Before / After
        before_label = Text("Before", font_size=24, color=SLATE)
        before_label.move_to(LEFT * 3 + UP * 2)
        before_score = Text("72%", font_size=72, color=SLATE,
                             font="Courier New")
        before_score.move_to(LEFT * 3 + UP * 0.5)

        after_label = Text("After", font_size=24, color=MINT)
        after_label.move_to(RIGHT * 3 + UP * 2)
        after_score = Text("91%", font_size=72, color=MINT,
                            font="Courier New")
        after_score.move_to(RIGHT * 3 + UP * 0.5)

        # Arrow between
        arrow = Arrow(LEFT * 1, RIGHT * 1, color=AMBER, stroke_width=3)
        arrow.move_to(UP * 0.7)

        self.play(FadeIn(before_label), Write(before_score), run_time=0.6)
        self.play(GrowArrow(arrow), run_time=0.4)
        self.play(FadeIn(after_label), Write(after_score), run_time=0.6)
        self.wait(0.5)

        # Stats
        stats = VGroup(
            Text("47 experiments", font_size=24, color=WHITE),
            Text("$2.34 total cost", font_size=24, color=AMBER_LIGHT),
            Text("overnight — unattended", font_size=24, color=SLATE),
        ).arrange(RIGHT, buff=1.2)
        stats.move_to(DOWN * 1.2)
        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in stats], lag_ratio=0.2),
            run_time=1,
        )
        self.wait(1.5)

        # Transition to end card
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
        self.wait(0.3)

        # End card
        logo_a = Text("anneal", font_size=64, weight=BOLD, color=WHITE)
        logo_a.move_to(UP * 0.5)
        tagline = Text("Autonomous optimization for any measurable artifact.",
                         font_size=22, color=SLATE)
        tagline.next_to(logo_a, DOWN, buff=0.4)

        self.play(FadeIn(logo_a, scale=0.8), run_time=0.8)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.5)
        self.wait(3)
