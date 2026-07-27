"""Course planning: checklist extraction, prerequisite planning, flowcharts.

See docs/course_planner.md. This package is the third pipeline in the project
(alongside ingestion and chat) and is deliberately independent of the LLM:
Architectural Decision AD-7 keeps course ordering in deterministic code.
"""
