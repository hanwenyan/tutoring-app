"""
Subject configurations for multi-subject tutoring support.

Each subject defines:
- icon: emoji for UI
- diagnostic_example: example for pedagogical framework
- topic_field_label: label for topic field in response structure
- latex_examples: subject-specific LaTeX notation examples
- real_world_examples: real-world connection examples
- ambiguity_example: example question for handling ambiguity
- kg_problem_examples: knowledge graph problem generation examples
- kg_example_nodes: example nodes for KG generation prompt
- greeting_examples: examples for default greeting
- safety_note: safety warnings (if applicable)
- redirect_example: example redirect for off-topic questions
"""

SUBJECTS = {
    "Chemistry": {
        "icon": "⚗️",
        "diagnostic_example": '"To solve this stoichiometry problem, start by writing the balanced equation. What reactants and products do you have?"',
        "topic_field_label": "chemistry topic being discussed",
        "latex_examples": """Common notation: subscripts for chemical formulas \(H_2O\), \(CO_2\), superscripts for charges \(Ca^{2+}\), \(SO_4^{2-}\), arrows for reactions \(\\to\), \(\\rightleftharpoons\).

For chemistry, write element symbols and formulas in regular math mode, NOT text mode:
- ✅ CORRECT: \(4Fe + 3O_2 \\to 2Fe_2O_3\)
- ❌ WRONG: \(4\\text{Fe} + 3\\text{O}_2 \\to 2\\text{Fe}_2\\text{O}_3\)

Keep subscripts/superscripts in math mode. Only use `\\text{}` for full words like units or labels:
- ✅ CORRECT: \(n = \\frac{m}{M} \\text{ (moles)}\)
- ❌ WRONG: \(\\text{n} = \\frac{\\text{m}}{\\text{M}}\)

For display equations: \[2Mg + O_2 \\to 2MgO\]""",
        "real_world_examples": "stoichiometry → baking ratios and cooking; gas laws → scuba diving and weather balloons; redox reactions → phone batteries and corrosion; equilibrium → soda carbonation",
        "ambiguity_example": '"Are you stuck on the math part (like unit conversion) or the chemistry concept (like what a limiting reactant is)?"',
        "kg_problem_examples": """
**Example (mole concept):** "How many atoms are in 2.5 moles of carbon? Show your work."

**Example (stoichiometry):** "If 10.0 g of magnesium reacts with oxygen ($2Mg + O_2 \\to 2MgO$), what mass of MgO is produced?"

**Example (gas laws):** "A balloon has 3.0 L of helium at 25°C. If heated to 100°C at constant pressure, what is the new volume?"

For chemistry: use real compounds, balanced equations, and SI units. K-8: simple mole/mass conversions. High school: multi-step stoichiometry, limiting reactants, percent yield. AP/College: thermodynamics, equilibrium calculations.""",
        "kg_example_nodes": """
{
  "mole_concept": {
    "description": "Understanding Avogadro's number and converting between moles, particles, and mass",
    "prerequisites": [],
    "estimated_hours": 2.5
  },
  "stoichiometry": {
    "description": "Using balanced equations to calculate reactant/product amounts",
    "prerequisites": ["mole_concept"],
    "estimated_hours": 4.0
  },
  "limiting_reactant": {
    "description": "Identifying which reactant runs out first in a reaction",
    "prerequisites": ["stoichiometry"],
    "estimated_hours": 3.0
  }
}
""",
        "greeting_examples": '"What\'s the difference between ionic and covalent bonds?" or "Help me balance this equation: Fe + O₂ → Fe₂O₃" or "I\'m confused about molarity"',
        "safety_note": "Never provide instructions for dangerous reactions, explosives, illegal drugs, or harmful substances. If asked, explain why the topic is unsafe and redirect to educational theory only.",
        "redirect_example": '"I\'m a chemistry tutor. Let\'s focus on chemistry concepts like reactions, bonding, or stoichiometry. What chemistry topic are you working on?"',
    },
    "Math": {
        "icon": "📐",
        "diagnostic_example": '"To solve this equation, what operation would undo adding 5? Walk me through your first step."',
        "topic_field_label": "math topic being discussed",
        "latex_examples": """Common notation: fractions \(\\frac{a}{b}\), exponents \(x^2\), \(2^{10}\), radicals \(\\sqrt{x}\), \(\\sqrt[3]{27}\), equations \(2x + 5 = 17\).

For algebra and calculus:
- ✅ CORRECT: \(f(x) = \\frac{x^2 - 4}{x + 2}\), \(\\int_0^1 x^2\\,dx\), \(\\lim_{x \\to 0} \\frac{\\sin x}{x}\)
- Use proper notation: \(\\triangle ABC\) for triangles, \(\\angle BAC\) for angles, \(\\overline{AB}\) for line segments

For inequalities: \(\\leq\), \(\\geq\), \(<\), \(>\)

Keep variables and numbers in math mode. Use `\\text{}` only for full words:
- ✅ CORRECT: \(A = \\pi r^2 \\text{ (area of circle)}\)
- ❌ WRONG: \(\\text{A} = \\text{π}\\text{r}^\\text{2}\)

For display equations: \[x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\]""",
        "real_world_examples": "linear equations → budgeting and unit pricing; geometry → architecture and design; statistics → polling and data analysis; calculus → physics motion and optimization",
        "ambiguity_example": '"Are you stuck on how to set up the equation, or on solving the algebra once you have it?"',
        "kg_problem_examples": """
**Example (fractions):** "Simplify $\\frac{12}{18}$ to lowest terms. Show your work."

**Example (linear equations):** "Solve for $x$: $3(x - 4) = 2x + 7$. Check your answer."

**Example (quadratic formula):** "Use the quadratic formula to solve $2x^2 - 5x - 3 = 0$."

**Example (geometry):** "A right triangle has legs of 5 cm and 12 cm. Find the hypotenuse."

For math: K-8 uses whole numbers and clean fractions. High school uses messier numbers, radicals, and multi-step problems. AP/SAT style: word problems, graphing, and applied context.""",
        "kg_example_nodes": """
{
  "arithmetic_operations": {
    "description": "Addition, subtraction, multiplication, division with whole numbers and decimals",
    "prerequisites": [],
    "estimated_hours": 3.0
  },
  "fractions": {
    "description": "Working with fractions: simplifying, adding, multiplying, dividing",
    "prerequisites": ["arithmetic_operations"],
    "estimated_hours": 4.0
  },
  "solving_linear_equations": {
    "description": "Isolating variables using inverse operations in one-variable equations",
    "prerequisites": ["arithmetic_operations"],
    "estimated_hours": 3.5
  },
  "pythagorean_theorem": {
    "description": "Using a² + b² = c² to solve for sides of right triangles",
    "prerequisites": ["arithmetic_operations"],
    "estimated_hours": 2.5
  }
}
""",
        "greeting_examples": '"Solve 2x + 5 = 17 step by step" or "Explain the Pythagorean theorem" or "What does the derivative mean?"',
        "safety_note": "",
        "redirect_example": '"I\'m a math tutor. Let\'s focus on math concepts like algebra, geometry, or calculus. What math topic are you working on?"',
    },
    "English": {
        "icon": "📚",
        "diagnostic_example": '"What is the main claim the author is making in this paragraph? Can you identify the sentence that states it most directly?"',
        "topic_field_label": "English/writing topic being discussed",
        "latex_examples": """For English, use minimal LaTeX. Prefer markdown:
- **Bold** for key terms and vocabulary
- *Italics* for book/article titles, emphasis, and foreign words
- > Blockquotes for passages and quoted text
- `Backticks` for grammar notation (like parts of speech labels)

Only use LaTeX for special cases:
- ✅ CORRECT: "The thesis is: *'Social media reshapes identity.'*"
- ✅ CORRECT: "**Metaphor**: comparing two unlike things without 'like' or 'as'"
- Avoid math notation unless discussing linguistic structure or poetry meter""",
        "real_world_examples": "thesis statements → debate and persuasion; grammar → clear professional writing; literary analysis → understanding media and news; rhetoric → advertising and speeches",
        "ambiguity_example": '"Are you stuck on understanding what the passage means, or on how to write your response?"',
        "kg_problem_examples": """
**Example (thesis writing):** "Read this prompt: 'Is social media good or bad for teens?' Write a clear thesis statement that takes a position."

**Example (grammar):** "Identify the error: 'Their going to the store later.' Explain why it's wrong and how to fix it."

**Example (paragraph analysis):** "Read this paragraph: [short passage]. What is the main idea? What evidence supports it?"

**Example (literary device):** "Find an example of a metaphor in this sentence: 'Time is a thief that steals our youth.' Explain what's being compared."

For English: K-8 uses simple sentences and clear examples. High school uses literary excerpts, rhetorical analysis, and essay structure. AP Lang/Lit style: argument analysis, synthesis, and close reading.""",
        "kg_example_nodes": """
{
  "parts_of_speech": {
    "description": "Identifying nouns, verbs, adjectives, adverbs, pronouns, and their functions",
    "prerequisites": [],
    "estimated_hours": 3.0
  },
  "sentence_structure": {
    "description": "Understanding subjects, predicates, clauses, and sentence types",
    "prerequisites": ["parts_of_speech"],
    "estimated_hours": 4.0
  },
  "thesis_writing": {
    "description": "Crafting clear, arguable thesis statements for essays",
    "prerequisites": ["sentence_structure"],
    "estimated_hours": 3.5
  },
  "paragraph_structure": {
    "description": "Topic sentences, evidence, analysis, and transitions in body paragraphs",
    "prerequisites": ["thesis_writing"],
    "estimated_hours": 3.0
  }
}
""",
        "greeting_examples": '"Help me write a thesis statement for my essay" or "What\'s the difference between their, there, and they\'re?" or "Analyze this paragraph for me"',
        "safety_note": "",
        "redirect_example": '"I\'m an English tutor. Let\'s focus on writing, grammar, reading comprehension, or literary analysis. What English topic are you working on?"',
    },
}

SUBJECT_NAMES = list(SUBJECTS.keys())
DEFAULT_SUBJECT = "Math"


def get_subject_config(name: str) -> dict:
    """Get subject configuration by name, with fallback to default."""
    return SUBJECTS.get(name, SUBJECTS[DEFAULT_SUBJECT])
