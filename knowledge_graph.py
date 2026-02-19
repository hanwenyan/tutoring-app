"""
Knowledge graph management for tutoring app.
Handles DAG validation, node state tracking, and LLM-generated curriculum.
"""

import graphlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage


def save_graph(graph: dict, path: Path):
    """Serialize knowledge graph to JSON and write to path."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(graph, f, indent=2)
        os.replace(tmp, path)
    except:
        os.unlink(tmp)
        raise


def load_graph(path: Path) -> dict | None:
    """Read knowledge graph from path. Returns None if no file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        st.warning("Knowledge graph was corrupted and has been reset.")
        return None


def get_graph_topology(graph: dict) -> list[str]:
    """Get or compute topological sort order of graph nodes. Cached in graph dict."""
    if "_topo_order" in graph:
        return graph["_topo_order"]

    nodes = graph.get("nodes", {})
    ts = graphlib.TopologicalSorter()
    for node_id, node_data in nodes.items():
        deps = node_data.get("deps", [])
        ts.add(node_id, *deps)

    topo_order = list(ts.static_order())
    graph["_topo_order"] = topo_order
    return topo_order


def validate_dag(nodes: dict) -> bool:
    """Validate that the knowledge graph is a DAG using topological sort. Returns True if valid."""
    try:
        ts = graphlib.TopologicalSorter()
        for node_id, node_data in nodes.items():
            deps = node_data.get("deps", [])
            ts.add(node_id, *deps)
        # This will raise CycleError if there's a cycle
        tuple(ts.static_order())
        return True
    except graphlib.CycleError:
        return False


def compute_node_state(node: dict) -> str:
    """Compute node state: mastered | failing | in_progress | untested."""
    mastery_level = node.get("mastery_level", 0)
    decay_score = node.get("decay_score", 1.0)
    times_tested = node.get("times_tested", 0)
    times_correct = node.get("times_correct", 0)

    # Green: mastered
    if mastery_level >= 3 and decay_score > 0.5:
        return "mastered"

    # Red: failing
    if times_tested > 0 and times_correct / times_tested < 0.5:
        return "failing"

    # Yellow: in progress or untested
    if times_tested > 0:
        return "in_progress"

    return "untested"


def get_next_node(graph: dict) -> str | None:
    """Return the first unmastered node with satisfied dependencies, in topological order."""
    nodes = graph.get("nodes", {})

    topo_order = get_graph_topology(graph)

    # Find first unmastered node with all deps mastered
    for node_id in topo_order:
        node = nodes[node_id]
        state = compute_node_state(node)

        if state != "mastered":
            # Check if all dependencies are mastered
            deps_mastered = all(
                compute_node_state(nodes[dep]) == "mastered"
                for dep in node.get("deps", [])
                if dep in nodes
            )
            if deps_mastered:
                return node_id

    return None


def is_node_locked(node_id: str, graph: dict) -> bool:
    """Check if a node is locked (has unmastered dependencies)."""
    nodes = graph.get("nodes", {})
    node = nodes.get(node_id)
    if not node:
        return True

    for dep in node.get("deps", []):
        if dep in nodes and compute_node_state(nodes[dep]) != "mastered":
            return True

    return False


def update_node_from_log(graph: dict, tutor_log: dict):
    """Update node state based on TUTOR_LOG verdict."""
    node_id = tutor_log.get("node")
    if not node_id or node_id not in graph.get("nodes", {}):
        return

    node = graph["nodes"][node_id]
    verdict = tutor_log.get("node_verdict", "not_assessed")

    # Update times_tested
    if verdict != "not_assessed":
        node["times_tested"] = node.get("times_tested", 0) + 1
        node["last_tested"] = datetime.now().isoformat()

    # Update mastery based on verdict
    if verdict == "mastered":
        node["mastery_level"] = 3
        node["times_correct"] = node.get("times_correct", 0) + 1
        node["problem_step"] = 3  # Completed all steps
        node["decay_score"] = 1.0  # Reset decay on mastery
    elif verdict == "progressing":
        # Advance problem_step
        current_step = node.get("problem_step", 0)
        node["problem_step"] = min(current_step + 1, 3)
        if node["problem_step"] >= 2:
            node["mastery_level"] = min(node.get("mastery_level", 0) + 1, 3)
        node["times_correct"] = node.get("times_correct", 0) + 1
    elif verdict == "struggling":
        # Drop back to atomic (step 1) if not already there
        if node.get("problem_step", 0) > 1:
            node["problem_step"] = 1


def build_graph_context(graph: dict, active_node: str | None) -> str:
    """Build the <knowledge_graph_state> block for system prompt injection."""
    if not graph or not active_node:
        return ""

    nodes = graph.get("nodes", {})
    if active_node not in nodes:
        return ""

    active = nodes[active_node]
    step = active.get("problem_step", 0)
    step_names = {0: "Untested", 1: "Atomic Problem", 2: "Variation Problem", 3: "Boss Problem"}

    lines = [
        "<knowledge_graph_state>",
        f"Current focus: {active_node} (Step {step}: {step_names.get(step, 'Unknown')})",
        f"- Description: {active.get('description', 'N/A')}",
        f"- Progress: {active.get('times_correct', 0)}/{active.get('times_tested', 0)} correct",
        "",
        "Dependencies:",
    ]

    # Show dependency status
    for dep in active.get("deps", []):
        if dep in nodes:
            dep_node = nodes[dep]
            state = compute_node_state(dep_node)
            decay = dep_node.get("decay_score", 1.0)
            status = "MASTERED" if state == "mastered" else state.upper()
            lines.append(f"- {dep}: {status} (decay: {decay:.2f})")

    # Show locked nodes
    locked = [nid for nid in nodes if is_node_locked(nid, graph) and compute_node_state(nodes[nid]) != "mastered"]
    if locked:
        lines.append("")
        lines.append("Locked (needs prereqs): " + ", ".join(locked[:5]))

    # Show review-due nodes (decay < 0.5)
    review_due = [nid for nid in nodes if compute_node_state(nodes[nid]) == "mastered" and nodes[nid].get("decay_score", 1.0) < 0.5]
    if review_due:
        lines.append("")
        lines.append("Review due: " + ", ".join(review_due[:3]))

    lines.append("</knowledge_graph_state>")
    return "\n".join(lines)


def generate_graph_prompt(subject: str, example_nodes: str = "") -> str:
    """Generate a prompt for creating a knowledge graph."""
    examples_block = f"\n\nExample nodes for {subject}:\n{example_nodes}" if example_nodes else ""
    return f"""Generate a knowledge graph for {subject} tutoring.

Create a JSON structure with 12-20 atomic, testable skills arranged as a dependency DAG. Each node should represent ONE specific skill that can be tested in isolation.

Requirements:
- Start with fundamental prerequisites
- Build up to complex applications
- Each node needs exactly 0-3 dependencies
- Use snake_case IDs (e.g., "mole_concept", "stoichiometry")
- Labels should be concise (3-6 words)
- Descriptions should be specific and testable{examples_block}

Return ONLY valid JSON in this exact format:
{{
  "subject": "{subject}",
  "nodes": {{
    "node_id": {{
      "label": "Node Label",
      "description": "Specific testable description",
      "deps": []
    }}
  }}
}}

Generate the complete graph now."""


def generate_graph(model, subject: str, example_nodes: str = "") -> dict | None:
    """Generate a knowledge graph using the LLM. Returns graph dict or None on error."""
    try:
        prompt = generate_graph_prompt(subject, example_nodes)
        messages = [
            SystemMessage(content="You are a curriculum design expert. Generate valid JSON only, no markdown formatting."),
            HumanMessage(content=prompt)
        ]

        response = model.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Strip markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        graph = json.loads(response_text)

        # Add default state fields to each node
        for node_id, node_data in graph.get("nodes", {}).items():
            node_data.setdefault("mastery_level", 0)
            node_data.setdefault("last_tested", None)
            node_data.setdefault("times_correct", 0)
            node_data.setdefault("times_tested", 0)
            node_data.setdefault("problem_step", 0)
            node_data.setdefault("decay_score", 1.0)

        # Validate DAG
        if not validate_dag(graph.get("nodes", {})):
            st.error("Generated graph contains cycles. Please try again.")
            return None

        return graph

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse graph JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Graph generation error: {e}")
        return None


def render_sidebar(provider_key: str, model_name: str, api_key: str, base_url: str, graph_file_path: Path, get_model_func):
    """Render the knowledge map section in the sidebar."""
    st.markdown("#### :blue[Knowledge Map]")

    if st.session_state.knowledge_graph is None:
        from subjects import get_subject_config
        active_subject = st.session_state.get("active_subject", "Chemistry")
        config = get_subject_config(active_subject)

        if st.button(f"Generate {active_subject} Map", use_container_width=True):
            if provider_key in ("nvidia", "gemini") and not api_key:
                st.error("Please provide your API Key first.")
            else:
                with st.spinner(f"Generating {active_subject} knowledge graph..."):
                    model = get_model_func(provider_key, model_name, api_key=api_key, base_url=base_url)
                    graph = generate_graph(model, active_subject, config.get("kg_example_nodes", ""))
                    if graph:
                        st.session_state.knowledge_graph = graph
                        save_graph(graph, graph_file_path)
                        # Auto-select first node
                        st.session_state.active_node = get_next_node(graph)
                        st.success(f"Created {len(graph['nodes'])} nodes!")
                        st.rerun()
    else:
        graph = st.session_state.knowledge_graph
        nodes = graph.get("nodes", {})

        # Progress bar
        mastered_count = sum(1 for n in nodes.values() if compute_node_state(n) == "mastered")
        total_count = len(nodes)
        progress = mastered_count / total_count if total_count > 0 else 0
        st.progress(progress, text=f"Progress: {mastered_count}/{total_count}")

        # Node list
        st.caption(f"**{graph.get('subject', 'Knowledge Graph')}**")

        # Build topological order
        topo_order = get_graph_topology(graph)

        # Display nodes
        for node_id in topo_order:
            node = nodes[node_id]
            state = compute_node_state(node)
            locked = is_node_locked(node_id, graph)

            # Choose emoji
            if state == "mastered":
                emoji = "🟢"
            elif state == "failing":
                emoji = "🔴"
            elif locked:
                emoji = "🔒"
            else:
                emoji = "🟡"

            # Active indicator
            indicator = " ●" if node_id == st.session_state.active_node else ""

            # Button label
            label = f"{emoji} {node['label']}{indicator}"

            # Step indicator for active node
            if node_id == st.session_state.active_node:
                step = node.get("problem_step", 0)
                if step > 0:
                    label += f" [{step}/3]"

            if st.button(label, key=f"node_{node_id}", disabled=locked, use_container_width=True):
                st.session_state.active_node = node_id
                # Inject a navigation prompt
                nav_msg = {
                    "role": "user",
                    "content": f"[NAVIGATE TO NODE: {node_id}]",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(nav_msg)
                st.rerun()

        # Review-due section
        review_due = [nid for nid in nodes if compute_node_state(nodes[nid]) == "mastered" and nodes[nid].get("decay_score", 1.0) < 0.5]
        if review_due:
            st.divider()
            st.caption("🔄 **Review Needed**")
            for nid in review_due[:3]:
                node = nodes[nid]
                if st.button(f"🔄 {node['label']}", key=f"review_{nid}", use_container_width=True):
                    st.session_state.active_node = nid
                    st.rerun()

        st.divider()
        if st.button("Reset Map", use_container_width=True):
            if graph_file_path.exists():
                graph_file_path.unlink()
            st.session_state.knowledge_graph = None
            st.session_state.active_node = None
            st.session_state.auto_start_needed = True
            st.rerun()


def process_tutor_response(tutor_log: dict, graph_file_path: Path):
    """Process tutor response and update knowledge graph state."""
    if st.session_state.knowledge_graph and tutor_log.get("node"):
        update_node_from_log(st.session_state.knowledge_graph, tutor_log)

        # Auto-advance if mastered
        if tutor_log.get("node_verdict") == "mastered":
            next_node = get_next_node(st.session_state.knowledge_graph)
            if next_node:
                st.session_state.active_node = next_node
                st.toast(f"✅ Mastered! Moving to: {st.session_state.knowledge_graph['nodes'][next_node]['label']}")

        # Apply decay to all mastered nodes
        for node in st.session_state.knowledge_graph.get("nodes", {}).values():
            if compute_node_state(node) == "mastered":
                node["decay_score"] = node.get("decay_score", 1.0) * 0.95
        save_graph(st.session_state.knowledge_graph, graph_file_path)
