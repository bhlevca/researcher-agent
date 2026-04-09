"""Tests for attachment reading: file context injection into agent prompts.

Verifies that uploaded file content is correctly extracted, wrapped with
explicit instructions, and passed into the crew topic so the LLM cannot
hallucinate that it "cannot read attachments".
"""

import asyncio

import aiosqlite
import pytest
import yaml
from pathlib import Path

from researcher.ingestion import get_file_context, init_files_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENTS_YAML = Path(__file__).resolve().parent.parent / "src" / "researcher" / "config" / "agents.yaml"
TASKS_YAML = Path(__file__).resolve().parent.parent / "src" / "researcher" / "config" / "tasks.yaml"


@pytest.fixture
def run(request):
    """Helper to run async coroutines in sync tests."""
    loop = asyncio.new_event_loop()
    yield loop.run_until_complete
    loop.close()


@pytest.fixture
def memory_db(tmp_path, run):
    """In-memory-like SQLite with the files table ready."""
    db_path = tmp_path / "test.db"

    async def _setup():
        db = await aiosqlite.connect(str(db_path))
        db.row_factory = aiosqlite.Row
        await init_files_table(db)
        return db

    db = run(_setup())
    yield db
    run(db.close())


async def _insert_file(db, file_id, user_id, filename, text):
    """Insert a fake file row for testing."""
    await db.execute(
        "INSERT INTO files (id, user_id, filename, extension, size, extracted_text, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, '2025-01-01T00:00:00Z')",
        (file_id, user_id, filename, ".txt", len(text), text),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# get_file_context format tests
# ---------------------------------------------------------------------------


class TestGetFileContext:
    """Verify get_file_context wraps content with explicit LLM instructions."""

    def test_empty_file_ids_returns_empty(self, memory_db, run):
        result = run(get_file_context(memory_db, "user1", []))
        assert result == ""

    def test_nonexistent_file_returns_empty(self, memory_db, run):
        result = run(get_file_context(memory_db, "user1", ["nonexistent"]))
        assert result == ""

    def test_single_file_has_header_and_footer(self, memory_db, run):
        run(_insert_file(memory_db, "f1", "user1", "paper.txt", "Hello world"))
        result = run(get_file_context(memory_db, "user1", ["f1"]))

        assert "=== ATTACHED FILE CONTENT" in result
        assert "=== END OF ATTACHED FILE CONTENT ===" in result

    def test_single_file_contains_extracted_text(self, memory_db, run):
        run(_insert_file(memory_db, "f1", "user1", "paper.txt", "Figure 1 shows velocity"))
        result = run(get_file_context(memory_db, "user1", ["f1"]))

        assert "Figure 1 shows velocity" in result
        assert "--- paper.txt ---" in result

    def test_context_contains_anti_hallucination_instructions(self, memory_db, run):
        run(_insert_file(memory_db, "f1", "user1", "doc.pdf", "Some PDF content"))
        result = run(get_file_context(memory_db, "user1", ["f1"]))

        # Must explicitly tell the LLM the content is readable
        assert "FULL TEXT extracted" in result
        assert "MUST use this text" in result
        assert "cannot read attachments" in result.lower() or "cannot read" in result

    def test_multiple_files_all_included(self, memory_db, run):
        run(_insert_file(memory_db, "f1", "user1", "ch1.txt", "Chapter 1 content"))
        run(_insert_file(memory_db, "f2", "user1", "ch2.txt", "Chapter 2 content"))
        result = run(get_file_context(memory_db, "user1", ["f1", "f2"]))

        assert "--- ch1.txt ---" in result
        assert "--- ch2.txt ---" in result
        assert "Chapter 1 content" in result
        assert "Chapter 2 content" in result

    def test_wrong_user_cannot_see_file(self, memory_db, run):
        run(_insert_file(memory_db, "f1", "user1", "secret.txt", "secret"))
        result = run(get_file_context(memory_db, "other_user", ["f1"]))
        assert result == ""


# ---------------------------------------------------------------------------
# Agent / Task YAML config tests
# ---------------------------------------------------------------------------


class TestAgentYamlAttachmentInstructions:
    """Verify the agent backstory tells the LLM about attached file content."""

    @pytest.fixture(autouse=True)
    def _load_yaml(self):
        with open(AGENTS_YAML) as f:
            self.config = yaml.safe_load(f)
        self.backstory = self.config["researcher"]["backstory"]

    def test_backstory_mentions_attached_files(self):
        assert "ATTACHED FILE" in self.backstory.upper()

    def test_backstory_says_can_read(self):
        assert "CAN read" in self.backstory or "can read" in self.backstory.lower()

    def test_backstory_forbids_cannot_read_claim(self):
        # Must warn the LLM not to say it can't read files
        backstory_upper = self.backstory.upper()
        assert "NEVER" in backstory_upper
        assert any(phrase in self.backstory for phrase in [
            "cannot read",
            "Cannot read",
            "CANNOT",
            "cannot access",
        ])

    def test_backstory_mentions_extracted_marker(self):
        # Must reference the actual marker used by get_file_context
        assert "ATTACHED FILE CONTENT" in self.backstory


class TestTaskYamlAttachmentInstructions:
    """Verify the task description references attached file content."""

    @pytest.fixture(autouse=True)
    def _load_yaml(self):
        with open(TASKS_YAML) as f:
            self.config = yaml.safe_load(f)
        self.description = self.config["research_task"]["description"]

    def test_task_mentions_attached_file_content(self):
        assert "ATTACHED FILE CONTENT" in self.description

    def test_task_says_read_directly(self):
        desc_lower = self.description.lower()
        assert "read" in desc_lower
        assert "primary source" in desc_lower or "directly" in desc_lower

    def test_task_forbids_cannot_access_claim(self):
        assert "cannot read" in self.description.lower() or "cannot access" in self.description.lower()


# ---------------------------------------------------------------------------
# Integration: file context flows into topic
# ---------------------------------------------------------------------------


class TestFileContextInTopic:
    """Verify that when file_ids are provided, the context block
    appears in the topic string that gets passed to build_crew."""

    def test_topic_starts_with_file_context(self, memory_db, run):
        """Simulates what /chat does: prepend file_context to the message."""
        run(_insert_file(memory_db, "f1", "user1", "results.csv", "x,y\n1,2\n3,4"))
        file_context = run(get_file_context(memory_db, "user1", ["f1"]))
        message = "Summarize the data in the attached file"
        topic = file_context + message

        # The topic the LLM sees must start with the file content block
        assert topic.startswith("=== ATTACHED FILE CONTENT")
        # The user's message is at the end
        assert topic.endswith(message)
        # The actual CSV data is in the middle
        assert "x,y" in topic
        assert "1,2" in topic

    def test_no_files_topic_is_just_message(self, memory_db, run):
        file_context = run(get_file_context(memory_db, "user1", []))
        message = "What is the weather?"
        topic = file_context + message
        assert topic == message
