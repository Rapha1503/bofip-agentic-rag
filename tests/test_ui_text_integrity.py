from __future__ import annotations

import unittest
from pathlib import Path


class UiTextIntegrityTests(unittest.TestCase):
    def test_public_ui_copy_keeps_french_accents_and_no_replacement_markers(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        qmark = chr(63)

        broken_fragments = [
            f"R{qmark}ponse",
            f"r{qmark}ponse",
            f"cl{qmark}",
            f"Cl{qmark}",
            f"d{qmark}sactiv",
            f"Mod{qmark}le",
            f"r{qmark}el",
            f"fran{qmark}ais",
            f"T{qmark}l{qmark}chargement",
            f"v{qmark}rification",
            f"R{qmark}sultat",
            f"pr{qmark}t",
            f"D{qmark}tail",
            f"{qmark}tape",
            f"requ{qmark}te",
            f"mod{qmark}le",
        ]
        for fragment in broken_fragments:
            self.assertNotIn(fragment, app_source)

        expected_fragments = [
            "Réponse sourcée",
            "clé",
            "Modèle",
            "français",
            "Journal de l'agent",
            "Justification sourcée",
            "Réponse avec réserve",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, app_source)

    def test_public_progress_is_under_submit_button_and_not_runtime_noise(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertNotIn("with st.form(", app_source)
        submit_pos = app_source.index("submit = st.button(")
        self.assertIn('"Analyse en cours" if is_running else "Analyser la question"', app_source)
        slot_pos = app_source.index("status_slot = st.empty()", submit_pos)
        form_end_pos = app_source.index("if submit:", slot_pos)

        self.assertLess(submit_pos, slot_pos)
        self.assertLess(slot_pos, form_end_pos)
        self.assertNotIn('emit_progress("Chargement du runtime"', app_source)
        self.assertNotIn('emit_progress("Runtime prêt"', app_source)
        self.assertNotIn('emit_progress("Agent lancé"', app_source)

    def test_progress_panel_displays_step_durations(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("progress-time", app_source)
        self.assertIn('"step_s": event_step_s', app_source)
        self.assertIn('"elapsed_s": event_elapsed_s', app_source)
        self.assertIn('payload.get("step_s")', app_source)
        self.assertIn('payload.get("elapsed_s")', app_source)


    def test_running_analysis_keeps_question_visible_in_dedicated_preview(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("last_submitted_query", app_source)
        self.assertIn("running_question_preview", app_source)
        self.assertNotIn('value=pending_analysis.get("query", "")', app_source)
        self.assertIn("st.session_state.last_submitted_query = query.strip()", app_source)

    def test_running_question_preview_state_is_synced_before_widget_render(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        sync_line = 'st.session_state["running_question_preview"] = pending_analysis.get("query", "")'

        self.assertIn(sync_line, app_source)
        sync_pos = app_source.index(sync_line)
        widget_pos = app_source.index('key="running_question_preview"')

        self.assertLess(sync_pos, widget_pos)

    def test_disabled_question_preview_text_stays_readable(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn(".stTextArea textarea:disabled", app_source)
        self.assertIn("-webkit-text-fill-color: var(--ink)", app_source)
        self.assertIn("opacity: 1 !important", app_source)

    def test_expander_header_uses_theme_color_without_hover(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('[data-testid="stExpander"] summary', app_source)
        self.assertIn("color: var(--burgundy-dark) !important", app_source)
        self.assertIn("fill: var(--burgundy-dark) !important", app_source)

    def test_ui_does_not_default_unknown_status_to_partial(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertNotIn('agent_result.get("answer_status", "partial")', app_source)
        self.assertNotIn('return mapping.get(status or "", ("partial"', app_source)
        self.assertIn('agent_result.get("answer_status", "insufficient_evidence")', app_source)

    def test_new_analysis_clears_previous_results_area(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("results_slot = st.empty()", app_source)
        results_slot_pos = app_source.index("results_slot = st.empty()")
        pending_pos = app_source.index("if pending_analysis:", results_slot_pos)
        empty_pos = app_source.index("results_slot.empty()", pending_pos)
        run_pos = app_source.index("latest_results = run_agent_query(", empty_pos)
        display_pos = app_source.index("with results_slot.container():", run_pos)

        self.assertLess(results_slot_pos, pending_pos)
        self.assertLess(empty_pos, run_pos)
        self.assertLess(run_pos, display_pos)

    def test_streamlit_stale_elements_are_neutralized_without_gray_overlay(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('[data-testid="stAppViewContainer"] [data-stale="true"]', app_source)
        self.assertIn('[data-testid="stAppViewContainer"] .staleElement', app_source)
        self.assertIn('[data-testid="stAppViewContainer"] [class*="stale"]', app_source)
        stale_block_pos = app_source.index('[data-testid="stAppViewContainer"] [data-stale="true"]')
        stale_block_end = app_source.index(".block-container", stale_block_pos)
        stale_block = app_source[stale_block_pos:stale_block_end]

        self.assertIn("display: none !important", stale_block)
        self.assertIn("filter: none !important", stale_block)
        self.assertNotIn("opacity: 0.33", stale_block)

    def test_provider_controls_use_reactive_model_and_provider_scoped_api_key(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("coerce_model_for_provider", app_source)
        self.assertIn('key="selected_model"', app_source)
        self.assertIn("_api_key_state_key(provider_id)", app_source)
        self.assertIn("_api_key_state_key(str(pending_provider_id))", app_source)
        self.assertNotIn('key="api_key_input"', app_source)
        self.assertIn("query_col, config_col = st.columns", app_source)
        config_pos = app_source.index("with config_col:")
        params_pos = app_source.index('st.markdown("### Paramètres")')
        self.assertLess(config_pos, params_pos)

    def test_provider_controls_are_not_disabled_during_analysis(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        config_start = app_source.index("with config_col:")
        query_start = app_source.index("with query_col:", config_start)
        config_block = app_source[config_start:query_start]

        self.assertNotIn("disabled=is_running", config_block)

    def test_submitted_api_key_is_kept_for_pending_analysis_only(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('"api_key": api_key if provider.get("requires_api_key", True) else ""', app_source)
        self.assertIn('pending_analysis.get("api_key", "")', app_source)
        self.assertIn('st.session_state.get(_api_key_state_key(str(pending_provider_id)), "")', app_source)

    def test_sidebar_is_collapsed_and_not_used_for_provider_controls(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('initial_sidebar_state="collapsed"', app_source)
        self.assertNotIn("sidebar-card", app_source)
        self.assertNotIn("linear-gradient(180deg, rgba(75,13,34,.98)", app_source)

    def test_clear_cache_button_is_under_provider_settings(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        sidebar_pos = app_source.index("with st.sidebar:")
        config_pos = app_source.index("with config_col:")
        cache_pos = app_source.index('st.button("Vider le cache"')
        reranker_pos = app_source.index("use_reranker = False", config_pos)

        self.assertLess(sidebar_pos, config_pos)
        self.assertGreater(cache_pos, config_pos)
        self.assertLess(cache_pos, reranker_pos)

    def test_result_cache_key_includes_status_logic_version(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("APP_RESULT_CACHE_VERSION", app_source)
        cache_version_pos = app_source.index("APP_RESULT_CACHE_VERSION")
        cache_material_pos = app_source.index("cache_material =")
        cache_key_pos = app_source.index("cache_key = hashlib.md5")
        self.assertLess(cache_version_pos, cache_key_pos)
        self.assertLess(cache_material_pos, cache_key_pos)
        self.assertIn("APP_RESULT_CACHE_VERSION", app_source[cache_material_pos:cache_key_pos])


if __name__ == "__main__":
    unittest.main()
