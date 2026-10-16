# Readable status colors

Hard difficulty labels used one fixed pink in both themes. Its declared color measured 2.77:1 against the light card surface. The label now uses the existing theme-aware red token: 5.83:1 against that same surface, and 6.37:1 against the dark card surface.

The light theme's success green is deepened from `#17815d` to `#116d4c`. This improves small success text on neutral surfaces and the green random-practice action on its tinted hover background. The latter's declared sRGB contrast increases from 3.81:1 to 4.84:1. Accepted-result actions keep their existing inverse text and hover behavior.

An isolated calculation parses the actual stylesheet and checks 26 text/background pairs across both themes: four neutral surfaces, random-practice normal/hover tints, success-result tint and accepted-result normal/hover states. All exceed the chosen 4.5:1 product target; the minimum is 4.839:1. This is source-color arithmetic, not browser visual QA or a claim about every composed, disabled or translucent state.

The change consists of two stylesheet declarations. Verification uses the existing frontend checks and local/hosted production builds with their existing budgets. The calculation is retained as release evidence. No production records are created.
