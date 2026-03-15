project = "UAV Mechanics & Implementation"
author = "Huseyn Kishiyev"
copyright = "2026, Huseyn Kishiyev"
release = "0.1.0"

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
	"dollarmath",
	"amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
